from itertools import count
import numpy as np

from Parser import read_passengers
from Basics import Event, Location, duration_between
from Control import Variables, compute_phi, passenger_data, expiration_data


def load_passengers(fraction=1, hours=18):
    passenger_df = read_passengers(fraction, hours)
    passenger_df['time'] = passenger_df['time']

    # Update values of phi before creating new passengers
    for t in passenger_df['time'].unique():
        UpdatePhi(t)

    # Create passenger events
    for p in passenger_df.itertuples():
        NewPassenger(p.time, Location(p.o_source, p.o_target, p.o_loc), Location(p.d_source, p.d_target, p.d_loc),
                     p.trip_distance, p.trip_duration, p.patience, p.VoT)

    return passenger_df['time'].max()


class Passenger:
    _ids = count(0)
    p_HV = {}
    p_AV = {}

    def __init__(self, time, origin, destination, trip_distance, trip_duration, patience, VoT, HVs=None, AVs=None):
        self.id = next(self._ids)
        self.requestTime = time
        self.origin = origin
        self.destination = destination
        self.tripDistance = trip_distance
        self.tripDuration = trip_duration
        self.expiredTime = time + patience
        self.VoT = VoT  # Value of time ($/hr)
        self.preferHV, self.fare = Passenger.choose_vehicle(self, HVs, AVs)
        if self.preferHV is not None:
            if self.preferHV:
                Passenger.p_HV[self.id] = self
            elif ~self.preferHV:
                Passenger.p_AV[self.id] = self

        # Record data ['p_id', 'request_t', 'trip_d', 'trip_t', 'VoT', 'fare', 'prefer_HV']
        passenger_data.append([self.id, self.requestTime, self.tripDistance, self.tripDuration, self.VoT, self.fare, self.preferHV])

    def __repr__(self):
        return 'Passenger_{}'.format(self.id)

    def min_wait_time(self, vehicles):
        nearest_time = float('inf')
        for vehicle in vehicles:
            nearest_time = min(duration_between(vehicle.loc, self.origin), nearest_time)
        return nearest_time

    def choose_vehicle(self, HV_v, AV_v):
        # Fare = Flag price + Unit price * Trip distance
        fare_HV = Variables.HVf1 + Variables.HVf2 * self.tripDistance
        fare_AV = Variables.AVf1 + Variables.AVf2 * self.tripDistance

        # TODO: When instantaneous demand > supply, provide accurate ETA. Currently capped min(ETA) = 20 min
        # Generalised cost = Fare + VoT / 3600 * (Estimation ratio * Time to the nearest vacant vehicle)
        GC_HV = fare_HV + self.VoT / 3600 * Variables.phiHV * min(self.min_wait_time(HV_v), 1200)
        GC_AV = fare_AV + self.VoT / 3600 * Variables.phiAV * min(self.min_wait_time(AV_v), 1200)

        # Logit choice based on GC (dis-utility) of vehicles
        _c = np.random.choice(['HV', 'AV', 'others'],
                              p=np.exp([-GC_HV, -GC_AV, -Variables.others_GC]) / (np.exp(-GC_HV) + np.exp(-GC_AV) + np.exp(-Variables.others_GC)))
        if _c == 'HV':
            return True, fare_HV  # Prefer HV
        elif _c == 'AV':
            return False, fare_AV  # Prefer AV
        else:
            return None, 0  # Prefer other modes

    def check_expiration(self, t):
        if t >= self.expiredTime and self.preferHV is not None:
            if self.preferHV:
                del Passenger.p_HV[self.id]
            elif ~self.preferHV:
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
