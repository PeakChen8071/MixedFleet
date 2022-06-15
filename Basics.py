import heapq
from itertools import count
import numpy as np
import networkx as nx

from Map import G, shortest_times

eventQueue = []
v_id = count(0)
p_id = count(0)


def compute_phi(waiting_passengers, vacant_vehicles):
    less = min(waiting_passengers, vacant_vehicles)
    more = max(waiting_passengers, vacant_vehicles)
    return max(1.0, np.exp(0.16979338 + 0.03466977 * less - 0.0140257 * more))


def random_loc():
    rng = np.random.default_rng()
    random_edge = rng.choice(G.edges)
    random_dist = np.random.uniform(0, G.edges[random_edge]['distance'])
    return Location(random_edge[0], random_edge[1], random_dist)


def path_between(from_loc, to_loc):
    if (from_loc.source is to_loc.source) and (from_loc.target is to_loc.target) and (from_loc.timeFromSource < to_loc.timeFromSource):
        return None  # There is no path node if both are on the same road, travelling downstream
    return nx.shortest_path(G, from_loc.target, to_loc.source, weight='duration')


def distance_between(from_loc, to_loc):
    path = path_between(from_loc, to_loc)
    if path:  # Cost is 0 if only 1 node exists
        cost = sum([G.edges[path[i], path[i + 1]]['distance'] for i in range(len(path) - 1)])
    else:  # The same road
        return to_loc.locFromSource - from_loc.locFromSource

    if from_loc.type != 'Intersection':
        cost += from_loc.locFromTarget
    if to_loc.type != 'Intersection':
        cost += to_loc.locFromSource
    return cost


def duration_between(from_loc, to_loc):
    if (from_loc.source is to_loc.source) and (from_loc.target is to_loc.target) and (from_loc.timeFromSource < to_loc.timeFromSource):
        return to_loc.timeFromSource - from_loc.timeFromSource
    else:
        cost = shortest_times.loc[from_loc.target, to_loc.source]  # Load pre-calculated travel times
        # cost = nx.shortest_path_length(G, from_loc.target, to_loc.source, weight='duration')

    if from_loc.type != 'Intersection':
        cost += from_loc.timeFromTarget
    if to_loc.type != 'Intersection':
        cost += to_loc.timeFromSource

    if cost == 0:
        return 1  # The minimum travel time is set to 1 sec, avoiding possible computational errors
    else:
        return cost


class Location:
    def __init__(self, source: int, target: int = None, loc_from_source: float = 0):
        self.type = 'Intersection'  # Assume a location is at its source intersection
        self.source = source
        self.target = source
        self.locFromSource = 0
        self.timeFromSource = 0
        self.locFromTarget = 0
        self.timeFromTarget = 0

        if target is not None:
            if loc_from_source == G.edges[source, target]['distance']:
                self.source = target
                self.target = target
            elif loc_from_source != 0:
                self.type = 'Road'
                self.target = target
                r_length = G.edges[source, target]['distance']
                r_travel_time = G.edges[source, target]['duration']
                self.locFromSource = loc_from_source
                self.timeFromSource = int(r_travel_time * self.locFromSource / r_length)
                self.locFromTarget = r_length - self.locFromSource
                self.timeFromTarget = r_travel_time - self.timeFromSource

    def __repr__(self):
        if self.type == 'Intersection':
            return 'nodes[{}]'.format(self.source)
        else:
            return 'edges{}_{}m'.format([self.source, self.target], round(self.locFromSource, 2))


class Event:
    """ Event priorities, the triggering order at the same tim
        1 : TripCompletion, PickUp, ChargerOn
        2 : NewEV
        3 : NewPassenger
        4 : Assign
    """

    def __init__(self, time, priority):
        self.time = time
        self.priority = priority
        heapq.heappush(eventQueue, self)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)
