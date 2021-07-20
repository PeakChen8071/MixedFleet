import networkx as nx

from Configuration import configs
from Basics import Event, duration_between
from Control import Variables, Statistics
from Demand import Passenger
from Supply import HVs, activeAVs, TripCompletion, ActivateAVs, DeactivateAVs, cruiseAV


# Bipartite matching which minimises the total dispatch trip duration
def bipartite_match(vacant_v, waiting_p):
    if (not vacant_v) | (not waiting_p):
        return None  # No matching if either set is empty

    # bipartite_edges = []
    bipartite_graph = nx.DiGraph()
    for p in waiting_p:
        for v in vacant_v:
            # _t = duration_between(v.loc, p.origin)
            # if _t <= 1200:  # Create bipartite edge if the pick-up time is within 30 min
            bipartite_graph.add_node(v, bipartite=0)
            bipartite_graph.add_node(p, bipartite=1)
            bipartite_graph.add_edge(v, p, duration=duration_between(v.loc, p.origin))

    results = None
    top_nodes = {n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 0}
    if not nx.is_empty(bipartite_graph):
        results = [(v, p, bipartite_graph.edges[v, p]['duration'])
                   for v, p in nx.bipartite.minimum_weight_full_matching(bipartite_graph, top_nodes=top_nodes, weight='duration').items()
                   if isinstance(p, Passenger)]

    return results  # List of tuples with assigned (vehicle, passenger)


def compute_assignment(t):
    for v in (HVs | activeAVs).values():
        # v.update_loc(t)  # Update vehicle location
        v.time = t  # Update vehicle time

    for p in (Passenger.p_HV | Passenger.p_AV).values():
        p.check_expiration(t)  # Remove expired passengers

    # Compute and return minimum weighting full bipartite matching
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
                v.occupiedTime = p.tripDuration  # Update vehicle occupied duration

                if meeting_t < Statistics.lastPassengerTime:
                    UpdateOccupied(meeting_t, v.is_HV, 1)
                if delivery_t < Statistics.lastPassengerTime:
                    UpdateOccupied(delivery_t, v.is_HV, -1)

                # Vehicle and passenger are assigned and removed from available dictionaries
                if v.is_HV:
                    del HVs[v.id]
                    del Passenger.p_HV[p.id]

                    # HVs receive incomes (wage = unit wage * trip duration)
                    v.income += Variables.unitWage * p.tripDuration
                else:
                    del activeAVs[v.id]
                    del Passenger.p_AV[p.id]

                    v.income += p.fare

                # Vehicle delivers passenger to passenger destination
                v.time = delivery_t
                v.loc = p.destination

                # Their next trip planning occurs after delivery trip completion
                v.nextTrip = TripCompletion(delivery_t, v, drop_off=True)

                # Record data ['v_id', 'p_id', 'dispatch_t', 'meeting_t', 'delivery_t']
                Statistics.assignment_data.append([v.id, p.id, self.time, meeting_t, delivery_t])

    def trigger(self, end=False):
        if not end:
            HV_match, AV_match = compute_assignment(self.time)
            self.match(HV_match)
            self.match(AV_match)


# Schedule assignment events, finish with assignment to catch all passengers
def schedule_assignment(endTime):
    roundEndTime = 0
    for t in range(0, endTime + configs['match_interval'], configs['match_interval']):
        Assign(t)
        roundEndTime = t
    Statistics.lastPassengerTime = roundEndTime


class UpdateOccupied(Event):
    def __init__(self, time, isHV, change):
        super().__init__(time, priority=0)
        self.isHV = isHV
        self.change = change

    def __repr__(self):
        return 'UpdateOccupied@t{} {}'.format(self.time, self.change)

    def trigger(self):
        if self.isHV:
            Statistics.HV_no += self.change
        else:
            Statistics.AV_no += self.change


# Dynamic AV fleet management: activation/deactivation
def manage_AVs():
    # TODO: Add argument handlers for optimisation control
    ActivateAVs(3600, 40)
    DeactivateAVs(7200, 5)
