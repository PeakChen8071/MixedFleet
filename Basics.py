import heapq
from itertools import count
import numpy as np
import networkx as nx

from Map import G


eventQueue = []


def random_loc():
    rng = np.random.default_rng()
    random_edge = rng.choice(G.edges)
    random_dist = np.random.uniform(0, G.edges[random_edge]['length'])
    return Location(random_edge[0], random_edge[1], random_dist)


def lonlat_to_loc(lon, lat, params=False):
    source = None
    target = None
    loc_from_source = 0
    nearest_dist = float('inf')

    for u, v in G.edges():
        # If passenger location coincides with an Intersection
        if (G.nodes[u]['pos'][0] == lon) & (G.nodes[u]['pos'][1] == lat):
            if params:
                return u, None, 0
            else:
                return Location(u)
        elif (G.nodes[v]['pos'][0] == lon) & (G.nodes[v]['pos'][1] == lat):
            if params:
                return v, None, 0
            else:
                return Location(v)

        if u == v:  # Skip loops in network
            continue

        vector = np.subtract(G.nodes[v]['pos'], G.nodes[u]['pos'])
        point = np.subtract((lon, lat), G.nodes[u]['pos'])
        unit_vector = vector / np.linalg.norm(vector)
        projected_dist = np.dot(point, unit_vector)
        if (projected_dist < 0) | (projected_dist > np.linalg.norm(vector)):
            continue  # Passenger must be within the line segment/edge
        perpendicular_dist = np.linalg.norm(point - projected_dist / np.linalg.norm(vector) * unit_vector)
        if perpendicular_dist <= nearest_dist:
            nearest_dist = perpendicular_dist
            source = u
            target = v
            loc_from_source = projected_dist / np.linalg.norm(vector) * G.edges[u, v]['length']

    if params:
        return source, target, loc_from_source
    else:
        return Location(source, target, loc_from_source)


def path_between(from_loc, to_loc):
    if (not isinstance(from_loc, Location)) | (not isinstance(to_loc, Location)):
        raise TypeError('Path must be calculated between 2 locations.')
    if (from_loc.source is to_loc.source) and (from_loc.target is to_loc.target) and (from_loc.locFromSource < to_loc.locFromSource):
        return None  # There is no path if both are on the same road, and vehicle is upstream to passenger
    return nx.shortest_path(G, from_loc.target, to_loc.source, weight='travel_time')


def distance_between(from_loc, to_loc):
    path = path_between(from_loc, to_loc)
    if path:
        cost = sum([G.edges[path[i], path[i + 1]]['length'] for i in range(len(path) - 1)])
    else:
        return to_loc.locFromSource - from_loc.locFromSource

    if from_loc.type != 'Intersection':
        cost += from_loc.locFromTarget
    if to_loc.type != 'Intersection':
        cost += to_loc.locFromSource
    return cost


def duration_between(from_loc, to_loc):
    if (from_loc.source is to_loc.source) and (from_loc.target is to_loc.target) and (from_loc.locFromSource < to_loc.locFromSource):
        return to_loc.timeFromSource - from_loc.timeFromSource
    else:
        cost = nx.shortest_path_length(G, from_loc.target, to_loc.source, weight='travel_time')

    if from_loc.type != 'Intersection':
        cost += from_loc.timeFromTarget
    if to_loc.type != 'Intersection':
        cost += to_loc.timeFromSource
    return cost


class Location:
    def __init__(self, source, target=None, loc_from_source=0):
        self.type = 'Intersection'  # Assume a location is at its source intersection
        self.source = source
        self.target = source
        self.locFromSource = 0
        self.timeFromSource = 0
        self.locFromTarget = 0
        self.timeFromTarget = 0

        if target is not None:
            if loc_from_source == G.edges[source, target]['length']:
                self.source = target
                self.target = target
            elif loc_from_source != 0:
                self.type = 'Road'
                self.target = target
                self.length = G.edges[source, target]['length']
                self.travelTime = G.edges[source, target]['travel_time']
                self.locFromSource = loc_from_source
                self.timeFromSource = int(self.travelTime * self.locFromSource / self.length)
                self.locFromTarget = self.length - self.locFromSource
                self.timeFromTarget = self.travelTime - self.timeFromSource

    def __repr__(self):
        if self.type == 'Intersection':
            return 'nodes[{}]'.format(self.source)
        else:
            return 'edges{}_{}m'.format([self.source, self.target], round(self.locFromSource, 2))


class Event:
    """ Event priorities, the triggering order at the same time

        0 : NewVehicle, CruiseTrip
        1 : UpdatePhi
        2 : NewPassenger
        3 : Assign
    """
    _ids = count(0)

    def __init__(self, time, priority):
        self.time = time
        self.priority = priority
        self.event_id = next(self._ids)
        heapq.heappush(eventQueue, self)

    def __lt__(self, other):
        return (self.time, self.priority, self.event_id) < (other.time, other.priority, other.event_id)
