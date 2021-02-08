import heapq
from itertools import count
import numpy as np

from Control import Variables
from Basics import Events, Location, distance_between
from Parser import read_passengers

passenger_df = read_passengers()
simulationEndTime = passenger_df['time'].max()


def load_passengers():
    for idx, p in passenger_df.iterrows():
        heapq.heappush(Events, (int(p['time']), 1, idx,
                                Location(p['origin_loc_source'],
                                         p['origin_loc_target'],
                                         p['origin_loc_distance']),
                                Location(p['destination_loc_source'],
                                         p['destination_loc_target'],
                                         p['destination_loc_distance']),
                                p['trip_distance'], p['trip_duration']))

    for i in passenger_df['time'].unique():  # Update phi before creating passengers for car choice
        heapq.heappush(Events, (int(i), 0))


class Passenger:
    _ids = count(0)
    p_w_HV = {}
    p_w_AV = {}
    cancelled = count(0)

    def __init__(self, time, origin, destination, trip_distance, trip_duration, HV_v=None, AV_v=None, patience=60):
        self.id = next(self._ids)
        self.expiredTime = time + int(np.random.normal(patience, patience / 10))
        self.origin = origin
        self.destination = destination
        self.tripDistance = trip_distance
        self.tripDuration = trip_duration
        self.preferredCar, self.fare = Passenger.choose_vehicle(self, HV_v, AV_v)
        if self.preferredCar is 'HV':
            Passenger.p_w_HV[self.id] = self
        elif self.preferredCar is 'AV':
            Passenger.p_w_AV[self.id] = self
        else:
            raise ValueError('Preferred vehicle type is unavailable.')

    def __repr__(self):
        return 'Passenger_{}'.format(self.id)

    def update(self, time):
        if time >= self.expiredTime:
            Passenger.cancelled = next(Passenger.cancelled)
            if self.preferredCar is 'HV':
                Passenger.p_w_HV.pop(self.id)
            elif self.preferredCar is 'AV':
                Passenger.p_w_AV.pop(self.id)
            else:
                raise KeyError('Passenger_{} is not in the waiting dictionary.'.format(self.id))
            del self

    def nearest_vehicle(self, vehicles):
        nearest_dist = float('inf')
        for vehicle in vehicles:
            nearest_dist = min(distance_between(vehicle.loc, self.origin), nearest_dist)
        return nearest_dist

    def choose_vehicle(self, HV_v, AV_v):
        valueOfTime = 50/3600  # assuming $50 per hour rate
        fare_HV = Variables.HVf1 + Variables.HVf2 * self.tripDistance
        fare_AV = Variables.AVf1 + Variables.AVf2 * self.tripDistance
        GC_HV = fare_HV + valueOfTime * Variables.phiHV * self.nearest_vehicle(HV_v)
        GC_AV = fare_AV + valueOfTime * Variables.phiAV * self.nearest_vehicle(AV_v)

        if np.random.rand() <= np.exp(GC_HV) / (np.exp(GC_HV) + np.exp(GC_AV)):
            return 'HV', fare_HV
        else:
            return 'AV', fare_AV


class PassengerEvent:
    priority = 1

    def __lt__(self, other):
        return self.priority < other.priority
