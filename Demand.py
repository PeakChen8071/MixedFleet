from itertools import count
from scipy.stats import truncnorm
import numpy as np

from Parser import read_passengers
from Basics import Event, Location, duration_between
from Control import Variables, compute_phi, passenger_data


def load_passengers(fraction=1, rows=None):
    passenger_df = read_passengers(fraction, rows)

    # Update values of phi before creating new passengers
    for t in passenger_df['time'].unique():
        UpdatePhi(t)

    # Create passenger events
    for idx, p in passenger_df.iterrows():
        NewPassenger(int(p['time']),
                     Location(p['origin_loc_source'],
                              p['origin_loc_target'],
                              p['origin_loc_distance']),
                     Location(p['destination_loc_source'],
                              p['destination_loc_target'],
                              p['destination_loc_distance']),
                     p['trip_distance'], int(p['trip_duration']))

    return passenger_df['time'].max()


class Passenger:
    _ids = count(0)
    p_HV = {}
    p_AV = {}

    def __init__(self, time, origin, destination, trip_distance, trip_duration, HVs=None, AVs=None):
        self.id = next(self._ids)
        self.startTime = time
        self.expiredTime = time + int(truncnorm.rvs(a=-10, b=10, loc=60, scale=6))  # TODO: patience
        self.VoT = int(truncnorm.rvs(a=-10, b=10, loc=50/3600, scale=5/3600))  # TODO: value of time
        self.origin = origin
        self.destination = destination
        self.tripDistance = trip_distance
        self.tripDuration = trip_duration
        self.preferHV, self.fare = Passenger.choose_vehicle(self, HVs, AVs)
        if self.preferHV:
            Passenger.p_HV[self.id] = self
        else:
            Passenger.p_AV[self.id] = self

        self.record_output()

    def __repr__(self):
        return 'Passenger_{}'.format(self.id)

    def min_wait_time(self, vehicles):
        durations = [duration_between(vehicle.loc, self.origin) for vehicle in vehicles]
        # TODO: When instantaneous demand > supply, provide approximate time to the 'nearest' vehicle
        durations.append(1200)  # Default nearest time from vehicles at 20 min, when there are no vacant vehicles
        return min(durations)

    def choose_vehicle(self, HV_v, AV_v):
        # Fare = Flag price + Unit price * Trip distance
        fare_HV = Variables.HVf1 + Variables.HVf2 * self.tripDistance
        fare_AV = Variables.AVf1 + Variables.AVf2 * self.tripDistance

        # Generalised cost = Fare + VoT * distanceRatio * TimeToNearestVehicle
        GC_HV = fare_HV + self.VoT * Variables.phiHV * self.min_wait_time(HV_v)
        GC_AV = fare_AV + self.VoT * Variables.phiAV * self.min_wait_time(AV_v)

        # Logit choice based on GC (dis-utility) of vehicles
        if np.random.rand() <= np.exp(-GC_HV) / (np.exp(-GC_HV) + np.exp(-GC_AV)):
            return True, fare_HV
        else:
            return False, fare_AV

    def check_expiration(self, t):
        if t >= self.expiredTime:
            passenger_data['expired'][self.id] = True

            if self.preferHV:
                del Passenger.p_HV[self.id]
            else:
                del Passenger.p_AV[self.id]

    def record_output(self):
        passenger_data['start_t'].update({self.id: self.startTime})
        passenger_data['expire_t'].update({self.id: self.expiredTime})
        passenger_data['trip_d'].update({self.id: self.tripDistance})
        passenger_data['fare'].update({self.id: self.fare})
        passenger_data['prefer_HV'].update({self.id: self.preferHV})
        passenger_data['expired'].update({self.id: False})


class UpdatePhi(Event):
    def __init__(self, time):
        super().__init__(time, priority=1)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'UpdatePhi@t{}'.format(self.time)

    def trigger(self, nHV, nAV):
        Variables.phiHV = compute_phi(len(Passenger.p_HV), nHV)
        Variables.phiAV = compute_phi(len(Passenger.p_AV), nAV)


class NewPassenger(Event):
    def __init__(self, time, *args):
        super().__init__(time, priority=2)
        self.args = args

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'Passenger@t{}'.format(self.time)

    def trigger(self, HVs, AVs):
        Passenger(self.time, self.args[0], self.args[1], self.args[2], self.args[3], HVs, AVs)
