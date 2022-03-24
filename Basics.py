import heapq
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import truncnorm

from Configuration import configs
from Map import G, shortest_times


eventQueue = []


# File is validated to include passenger attributes for future simulations.
# Similar to using a random seed which maintains stochastic attributes over difference simulations.
def validate_passengers(passenger_file):
    cols = pd.read_csv(passenger_file, nrows=1).columns
    if 'patience' not in cols:
        print('Injecting passenger attributes...')
        df = pd.read_csv(configs["passenger_file"])
        df['tpep_pickup_datetime'] = (pd.to_datetime(df['tpep_pickup_datetime']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

        # Calculate trip properties for access in future simulations
        df['trip_distance'] = df.apply(lambda x: distance_between(Location(x['o_source'], x['o_target'], x['o_loc']),
                                                                  Location(x['d_source'], x['d_target'], x['d_loc'])), axis=1)
        df['trip_duration'] = df.apply(lambda x: duration_between(Location(x['o_source'], x['o_target'], x['o_loc']),
                                                                  Location(x['d_source'], x['d_target'], x['d_loc'])), axis=1)

        # Random patience time (sec) ~ Normal(60, 6^2) bounded by [30, 90]
        df['patience'] = truncnorm.rvs(a=-5, b=5, loc=60, scale=6, size=df.shape[0]).astype(int)

        # Random VoT ($/hr) ~ Normal(32, 3.2^2) bounded by [22, 38], rounded to int. (NYC HDM, 2018 household income)
        # VoT might be underestimated for Manhattan which is a relatively high-income area (Ulak, et al., 2020)
        df['VoT'] = truncnorm.rvs(a=-3.125, b=1.875, loc=32, scale=3.2, size=df.shape[0])
        df['VoT'] = df['VoT'].round(2)  # Round to the nearest cents for readability

        # Individual utility parameters are assumed to follow truncated normal distributions (professional knowledge)
        df['AV_const'] = truncnorm.rvs(a=-1, b=1, loc=0, scale=0.5, size=df.shape[0])
        df['HV_const'] = truncnorm.rvs(a=-1, b=1, loc=0, scale=0.5, size=df.shape[0])
        df['AV_coef_fare'] = truncnorm.rvs(a=-1, b=1, loc=0.2, scale=0.2, size=df.shape[0])
        df['HV_coef_fare'] = truncnorm.rvs(a=-1, b=1, loc=0.2, scale=0.2, size=df.shape[0])
        df['AV_coef_time'] = truncnorm.rvs(a=-1, b=1, loc=0.05, scale=0.05, size=df.shape[0])
        df['HV_coef_time'] = truncnorm.rvs(a=-1, b=1, loc=0.05, scale=0.05, size=df.shape[0])

        # Write back to passenger file with injected attributes
        df.sort_values('tpep_pickup_datetime').to_csv(configs["passenger_file"], index=False)
        print('Attribute injection is completed.')


def random_loc():
    rng = np.random.default_rng()
    random_edge = rng.choice(G.edges)
    random_dist = np.random.uniform(0, G.edges[random_edge]['distance'])
    return Location(random_edge[0], random_edge[1], random_dist)


def path_between(from_loc, to_loc):
    assert (isinstance(from_loc, Location) and isinstance(to_loc, Location)), 'Path must be calculated between 2 locations.'
    if (from_loc.source is to_loc.source) and (from_loc.target is to_loc.target) and (from_loc.timeFromSource < to_loc.timeFromSource):
        return None  # There is no path if both are on the same road, and vehicle is upstream to passenger
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
        # cost = nx.shortest_path_length(G, from_loc.target, to_loc.source, weight='duration')
        cost = shortest_times.loc[from_loc.target, to_loc.source]

    if from_loc.type != 'Intersection':
        cost += from_loc.timeFromTarget
    if to_loc.type != 'Intersection':
        cost += to_loc.timeFromSource
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
                r_travelTime = G.edges[source, target]['duration']
                self.locFromSource = loc_from_source
                self.timeFromSource = int(r_travelTime * self.locFromSource / r_length)
                self.locFromTarget = r_length - self.locFromSource
                self.timeFromTarget = r_travelTime - self.timeFromSource

    def __repr__(self):
        if self.type == 'Intersection':
            return 'nodes[{}]'.format(self.source)
        else:
            return 'edges{}_{}m'.format([self.source, self.target], round(self.locFromSource, 2))


class Event:
    """ Event priorities, the triggering order at the same time

        0 : NewHV, ActivateAVs/DeactivateAVs
        1 : TripCompletion
        2 : UpdatePhi
        3 : NewPassenger
        4 : UpdateStates
        5 : Assign
        6 : MPC
    """

    def __init__(self, time, priority):
        self.time = int(time)
        self.priority = int(priority)
        heapq.heappush(eventQueue, self)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)


class Electricity:
    max_SoC = 50  # unit: kWh
    min_SoC = 10  # unit: kWh
    charge_rate = 7  # unit: kW
    consumption_rate = 5  # unit: kW
