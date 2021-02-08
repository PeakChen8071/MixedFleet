import math

from Basics import *
from Demand import Passenger
from Supply import HV, AV
from Control import Variables


def update_time(t):
    for pHV in Passenger.p_w_HV.values():
        pHV.update(t)
    for pAV in Passenger.p_w_AV.values():
        pAV.update(t)

    for hv in HV.HV_v.values():
        hv.update(t)
    for av in AV.AV_v.values():
        av.update(t)


# Exponential regression function which approximates the ratio of Actual matched distance to Nearest vehicle distance
def update_phi():
    x1 = min(len(HV.HV_v), len(Passenger.p_w_HV))
    y1 = max(len(HV.HV_v), len(Passenger.p_w_HV))
    x2 = min(len(AV.AV_v), len(Passenger.p_w_AV))
    y2 = max(len(AV.AV_v), len(Passenger.p_w_AV))
    Variables.phiHV = math.exp(0.16979338 + 0.03466977 * x1 - 0.0140257 * y1)
    Variables.phiAV = math.exp(0.16979338 + 0.03466977 * x2 - 0.0140257 * y2)


# Bipartite matching which minimises the total dispatch trip duration
def bipartite_match(vacant_v, waiting_p):
    if (not vacant_v) | (not waiting_p):
        return None  # No matching if either set is empty
    bipartite_edges = []
    for p in waiting_p.values():
        for v in vacant_v.values():
            bipartite_edges.append((v, p, {'trip_distance': distance_between(v.loc, p.origin)}))
            # bipartite_edges.append((v, p, {'trip_duration': duration_between(v.loc, p.origin)}))
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(vacant_v.values(), bipartite=0)
    bipartite_graph.add_nodes_from(waiting_p.values(), bipartite=1)
    bipartite_graph.add_edges_from(bipartite_edges)
    return nx.bipartite.minimum_weight_full_matching(bipartite_graph, weight='trip_distance')


def compute_assignment(t):
    update_time(t)
    HV_match = bipartite_match(HV.HV_v, Passenger.p_w_HV)
    AV_match = bipartite_match(AV.AV_v, Passenger.p_w_AV)
    return HV_match, AV_match


class Trip:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.beginTime = vehicle.time
        if self.vehicle.state == 'assigned':
            self.purpose = 'dispatch'
            self.destination = vehicle.withPassenger.origin
        elif self.vehicle.state == 'occupied':
            self.purpose = 'delivery'
            self.destination = vehicle.withPassenger.destination
        else:
            raise ValueError('Vehicle must be assigned or occupied.')

        self.passenger = None
        self.fare = None
        self.path = None
        self.distance = None
        self.duration = None
        self.arrivalTime = None

        if self.purpose == 'dispatch':
            self.path = path_between(vehicle.loc, vehicle.withPassenger.origin)
            self.distance = distance_between(vehicle.loc, self.destination)
            self.duration = duration_between(vehicle.loc, self.destination)
            self.arrivalTime = self.beginTime + self.duration
        elif self.purpose == 'delivery':
            self.path = path_between(vehicle.withPassenger.origin, vehicle.withPassenger.destination)
            self.distance = distance_between(vehicle.withPassenger.origin, vehicle.withPassenger.destination)
            self.duration = duration_between(vehicle.withPassenger.origin, vehicle.withPassenger.destination)
            self.arrivalTime = self.beginTime + self.duration
        else:
            raise ValueError('Incorrect destination for trip purpose: {}'.format(self.purpose))



