import networkx as nx

from Configuration import configs
from Basics import Event, duration_between
from Control import assignment_data, Variables
from Demand import Passenger
from Supply import HVs, activeAVs, TripCompletion, ActivateAVs, DeactivateAVs, cruiseAV


# Bipartite matching which minimises the total dispatch trip duration
def bipartite_match(vacant_v, waiting_p):
    if (not vacant_v) | (not waiting_p):
        return None  # No matching if either set is empty

    bipartite_edges = []
    for p in waiting_p:
        for v in vacant_v:
            bipartite_edges.append((v, p, {'duration': duration_between(v.loc, p.origin)}))
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(vacant_v, bipartite=0)
    bipartite_graph.add_nodes_from(waiting_p, bipartite=1)
    bipartite_graph.add_edges_from(bipartite_edges)

    results = []
    for v, p in nx.bipartite.minimum_weight_full_matching(bipartite_graph, weight='duration').items():
        if isinstance(p, Passenger):
            results.append((v, p, bipartite_graph.edges[v, p]['duration']))
    return results  # List of tuples with assigned (vehicle, passenger)


def compute_assignment(t):
    for v in (HVs | activeAVs).values():
        # v.update_loc(t)  # Update vehicle location
        v.time = t  # Update vehicle time

    for p in (Passenger.p_HV | Passenger.p_AV).values():
        p.check_expiration(t)  # Remove expired passengers

    # Compute and return minimum weighting maximal bipartite matching
    HV_match = bipartite_match(HVs.values(), Passenger.p_HV.values())
    AV_match = bipartite_match(activeAVs.values(), Passenger.p_AV.values())
    return HV_match, AV_match


class Assign(Event):
    def __init__(self, time):
        super().__init__(time, priority=4)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'Assignment@t{}'.format(self.time)

    def match(self, match_results):
        if match_results:
            for m in match_results:
                v = m[0]  # Assigned vehicle
                p = m[1]  # Assigned passenger
                meeting_t = self.time + m[2]  # Timestamp of meeting
                delivery_t = meeting_t + p.tripDuration  # Timestamp of drop-off
                v.occupiedTime += p.tripDuration  # Update vehicle occupied duration

                # Vehicle and passenger are assigned and removed from available dictionaries
                if v.is_HV:
                    del HVs[v.id]
                    del Passenger.p_HV[p.id]

                    # HVs receive incomes (wage = unit wage * trip duration)
                    v.income += Variables.unitWage * p.tripDuration
                else:
                    del activeAVs[v.id]
                    del Passenger.p_AV[p.id]

                # Vehicle delivers passenger to passenger destination
                v.time = delivery_t
                v.loc = p.destination

                # Their next trip planning occurs after delivery trip completion
                v.nextTrip = TripCompletion(delivery_t, v, drop_off=True)

                # Record data ['v_id', 'p_id', 'dispatch_t', 'meeting_t', 'delivery_t']
                assignment_data.append([v.id, p.id, self.time, meeting_t, delivery_t])

    def trigger(self, end=False):
        if not end:
            HV_match, AV_match = compute_assignment(self.time)
            self.match(HV_match)
            self.match(AV_match)


# Schedule assignment events, finish with assignment to catch all passengers
def schedule_assignment(endTime):
    for t in range(0, endTime, configs['match_interval']):
        Assign(t)


# Dynamic AV fleet management: activation/deactivation
def manage_AVs():
    # TODO: Add argument handlers for optimisation control
    ActivateAVs(3600, 40)
    DeactivateAVs(7200, 5)
