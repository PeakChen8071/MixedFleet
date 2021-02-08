import numpy as np

from Map import nx, utm, G

Events = []


def random_loc():
    rng = np.random.default_rng()
    random_edge = rng.choice(G.edges)
    random_dist = round(np.random.uniform(0, G.edges[random_edge]['length']), 2)
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
            del projected_dist
            continue  # Passenger must be within the line segment (i.e. edge)
        perpendicular_dist = np.linalg.norm(point - projected_dist / np.linalg.norm(vector) * unit_vector)
        if perpendicular_dist <= nearest_dist:
            nearest_dist = perpendicular_dist
            source = u
            target = v
            loc_from_source = round(projected_dist / np.linalg.norm(vector) * G.edges[u, v]['length'], 2)

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
    def __init__(self, source, target=None, loc_from_source=0):
        if (target is not None) & (loc_from_source != 0):
            if loc_from_source != G.edges[source, target]['length']:
                self.type = 'Road'
                self.source = source
                self.target = target
                self.length = G.edges[source, target]['length']
                self.travelTime = G.edges[source, target]['travel_time']
                if (loc_from_source > 0) & (loc_from_source < self.length):
                    self.locFromSource = loc_from_source
                    self.timeFromSource = int(self.travelTime * self.locFromSource / self.length)
                    self.locFromTarget = self.length - self.locFromSource
                    self.timeFromTarget = self.travelTime - self.timeFromSource
                else:
                    raise ValueError('Value of loc_from_source is invalid.')
            else:
                self.type = 'Intersection'
                self.source = target
                self.target = target
                self.locFromSource = None
                self.locFromTarget = None
        else:
            self.type = 'Intersection'
            self.source = source
            self.target = source
            self.locFromSource = None
            self.locFromTarget = None

    def __repr__(self):
        if self.type is 'Intersection':
            return 'nodes[{}]'.format(self.source)
        else:
            return 'edges{}_{}m'.format([self.source, self.target], self.locFromSource)

    def is_intersection(self):
        if self.type is 'Intersection':
            return True
        else:
            return False
