from itertools import count
import numpy as np

from Parser import read_passengers
from Basics import Event, Location, duration_between
from Control import Variables, compute_phi, passenger_data, expiration_data


def load_passengers(fraction=1, rows=None):
    passenger_df = read_passengers(fraction, rows)

    # Update values of phi before creating new passengers
    for t in passenger_df['time'].unique():
        UpdatePhi(t)

    # Create passenger events
    for p in passenger_df.itertuples():
        NewPassenger(p.time,
                     Location(p.origin_loc_source,
                              p.origin_loc_target,
                              p.origin_loc_distance),
                     Location(p.destination_loc_source,
                              p.destination_loc_target,
                              p.destination_loc_distance),
                     p.trip_distance, p.trip_duration,
                     p.patience, p.VoT)

    return passenger_df['time'].max()


class Passenger:
    _ids = count(0)
    p_HV = {}
    p_AV = {}

    def __init__(self, time, origin, destination, trip_distance, trip_duration, patience, VoT, HVs=None, AVs=None):
        self.id = next(self._ids)
        self.startTime = time
        self.origin = origin
        self.destination = destination
        self.tripDistance = trip_distance
        self.tripDuration = trip_duration
        self.expiredTime = time + patience
        self.VoT = VoT  # Value of time ($/hr)
        self.preferHV, self.fare = Passenger.choose_vehicle(self, HVs, AVs)
        if self.preferHV:
            Passenger.p_HV[self.id] = self
        else:
            Passenger.p_AV[self.id] = self

        # Record data ['p_id', 'start_t', 'trip_d', 'VoT', 'fare', 'prefer_HV']
        passenger_data.append([self.id, self.startTime, self.tripDistance, self.VoT, self.fare, self.preferHV])

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

        # TODO: Include IVTT cost as part of GC?
        # Generalised cost = Fare + VoT / 3600 * distanceRatio * TimeToNearestVehicle
        GC_HV = fare_HV + self.VoT / 3600 * Variables.phiHV * self.min_wait_time(HV_v)
        GC_AV = fare_AV + self.VoT / 3600 * Variables.phiAV * self.min_wait_time(AV_v)

        # Logit choice based on GC (dis-utility) of vehicles
        if np.random.rand() <= np.exp(-GC_HV) / (np.exp(-GC_HV) + np.exp(-GC_AV)):
            return True, fare_HV  # Prefer HV
        else:
            return False, fare_AV  # Prefer AV

    def check_expiration(self, t):
        if t >= self.expiredTime:
            if self.preferHV:
                del Passenger.p_HV[self.id]
            else:
                del Passenger.p_AV[self.id]

            # Record data ['p_id', 'expire_t']
            expiration_data.append([self.id, self.expiredTime])


class UpdatePhi(Event):
    def __init__(self, time):
        super().__init__(time, priority=2)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'UpdatePhi@t{}'.format(self.time)

    def trigger(self, nHV, nAV):
        Variables.phiHV = compute_phi(len(Passenger.p_HV), nHV)
        Variables.phiAV = compute_phi(len(Passenger.p_AV), nAV)


class NewPassenger(Event):
    def __init__(self, time, *args):
        super().__init__(time, priority=3)
        self.args = args

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'Passenger@t{}'.format(self.time)

    def trigger(self, HVs, AVs):
        Passenger(self.time, *self.args, HVs, AVs)
