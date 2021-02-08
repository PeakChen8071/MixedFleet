import numpy as np
import networkx as nx
import utm

from Map import G

Events = []


def random_loc():
    rng = np.random.default_rng()
    random_edge = rng.choice(G.edges)
    random_dist = np.random.uniform(0, G.edges[random_edge]['length'])
    return Location(random_edge[0], random_edge[1], random_dist)


def lonlat_to_loc(lon, lat):
    source = None
    target = None
    loc_from_source = None
    nearest_dist = float('inf')

    for u, v in G.edges():
        if u == v:
            continue  # Skip loops in network
        # If passenger location coincides with an Intersection
        if (G.nodes[u]['pos'][0] == lon) & (G.nodes[u]['pos'][1] == lat):
            return Location(u)
        elif (G.nodes[v]['pos'][0] == lon) & (G.nodes[v]['pos'][1] == lat):
            return Location(v)

        vector = np.array(G.nodes[v]['xy']) - np.array(G.nodes[u]['xy'])
        point = np.array(utm.from_latlon(lat, lon)[:2]) - np.array(G.nodes[u]['xy'])
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

    return Location(source, target, loc_from_source)


def path_between(from_loc, to_loc, weight='travel_time'):
    if (not isinstance(from_loc, Location)) | (not isinstance(to_loc, Location)):
        raise TypeError('Path must be calculated between 2 locations.')
    return nx.shortest_path(G, from_loc.target, to_loc.source, weight=weight)


def distance_between(from_loc, to_loc, weight='travel_time'):
    path = path_between(from_loc, to_loc, weight=weight)
    cost = sum([G.edges[path[i], path[i + 1]]['length'] for i in range(len(path) - 1)])
    if not from_loc.is_intersection():
        cost += from_loc.locFromTarget
    if not to_loc.is_intersection():
        cost += to_loc.locFromSource
    return cost


def duration_between(from_loc, to_loc, weight='travel_time'):
    path = path_between(from_loc, to_loc, weight=weight)
    cost = sum([G.edges[path[i], path[i + 1]]['travel_time'] for i in range(len(path) - 1)])
    if not from_loc.is_intersection():
        cost += from_loc.timeFromTarget
    if not to_loc.is_intersection():
        cost += to_loc.timeFromSource
    return cost


class Location:
    def __init__(self, source, target=None, loc_from_source=None):
        self.type = 'Intersection'
        self.source = source
        self.target = None
        self.locFromSource = None
        self.timeFromSource = None
        self.locFromTarget = None
        self.timeFromTarget = None
        if (target is not None) and (loc_from_source is not None):
            if loc_from_source == G.edges[source, target]['length']:
                self.source = target
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
        if self.type is 'Intersection':
            return 'nodes[{}]'.format(self.source)
        else:
            return 'edges{}_{}m'.format([self.source, self.target], round(self.locFromSource, 2))

    def is_intersection(self):
        return self.type is 'Intersection'
