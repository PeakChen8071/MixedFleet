import networkx as nx

from Basics import Event, duration_between, distance_between
from Control import assignment_data
from Demand import Passenger
from Supply import HVs, AVs, CruiseTrip


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

    return results


def compute_assignment(t):
    for v in (HVs | AVs).values():
        v.update_loc(t)  # Update vehicle locations

    for p in (Passenger.p_HV | Passenger.p_AV).values():
        p.check_expiration(t)  # Remove expired passengers

    HV_match = bipartite_match(HVs.values(), Passenger.p_HV.values())
    AV_match = bipartite_match(AVs.values(), Passenger.p_AV.values())
    return HV_match, AV_match


class Assign(Event):
    def __init__(self, time):
        super().__init__(time, priority=3)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'Assignment@t{}'.format(self.time)

    def match(self, match_results):
        if match_results is not None:
            for m in match_results:
                v = m[0]
                p = m[1]
                dispatch_t = self.time + m[2]
                delivery_t = dispatch_t + p.tripDuration

                if v.is_HV:
                    del HVs[v.id]
                    del Passenger.p_HV[p.id]
                else:
                    del AVs[v.id]
                    del Passenger.p_AV[p.id]

                v.nextCruise = CruiseTrip(delivery_t, v, drop_off=True)

                # Assignment results output
                trip_id = next(self._ids)
                assignment_data['v_id'].update({trip_id: v.id})
                assignment_data['p_id'].update({trip_id: p.id})
                assignment_data['assignment_t'].update({trip_id: self.time})
                assignment_data['dispatch_t'].update({trip_id: dispatch_t})
                assignment_data['delivery_t'].update({trip_id: delivery_t})
                assignment_data['dispatch_d'].update({trip_id: distance_between(v.loc, p.origin)})

    def trigger(self):
        HV_match, AV_match = compute_assignment(self.time)
        self.match(HV_match)
        self.match(AV_match)
