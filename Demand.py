from itertools import count
import numpy as np
import pandas as pd

from Configuration import configs
from Parser import read_passengers
from Basics import Event, Location, duration_between
from Control import Statistics, Parameters, Variables, compute_phi


def load_passengers(fraction=1, hours=18):
    passenger_df = read_passengers(fraction, hours)
    lastPaxTime = passenger_df['time'].max()

    # Set demand prediction values
    bins = [i for i in range(0, lastPaxTime, configs['MPC_prediction_interval'])]
    Variables.histDemand = passenger_df['time'].groupby(pd.cut(passenger_df['time'], bins)).count().tolist()

    # Update values of phi before creating new passengers
    for t in passenger_df['time'].unique():
        UpdatePhi(t)

    # Create passenger events
    for p in passenger_df.itertuples():
        NewPassenger(p.time, Location(p.o_source, p.o_target, p.o_loc), Location(p.d_source, p.d_target, p.d_loc),
                     p.trip_distance, p.trip_duration, p.patience, p.VoT)

    Statistics.simulationEndTime = lastPaxTime
    Statistics.lastPassengerTime = lastPaxTime


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
        self.VoT = VoT / 3600 # Value of time ($/sec)
        self.preferHV, self.fare = Passenger.choose_vehicle(self, HVs, AVs)
        if self.preferHV is not None:
            if self.preferHV:
                Passenger.p_HV[self.id] = self
            elif ~self.preferHV:
                Passenger.p_AV[self.id] = self

        # Record statistics
        Statistics.passenger_data = Statistics.passenger_data.append({'p_id': self.id,
                                                                      'request_t': self.requestTime,
                                                                      'trip_d': self.tripDistance,
                                                                      'trip_t': self.tripDuration,
                                                                      'VoT': self.VoT,
                                                                      'fare': self.fare,
                                                                      'prefer_HV': self.preferHV}, ignore_index=True)

    def __repr__(self):
        return 'Passenger_{}'.format(self.id)

    # TODO: Optimise nearest time calls to reduce repetition?
    def min_wait_time(self, vehicles):
        nearest_time = float('inf')
        for vehicle in vehicles:
            nearest_time = min(duration_between(vehicle.loc, self.origin), nearest_time)
        return nearest_time

    def choose_vehicle(self, HV_v, AV_v):
        # Fare = Unit price * Trip duration
        fare_HV = Variables.HV_unitFare / 3600 * self.tripDuration
        fare_AV = Variables.AV_unitFare / 3600 * self.tripDuration

        # TODO: When instantaneous demand > supply, give accurate ETA. Now capped ETA = 20 min, use search radius
        # Generalised cost = Fare + VoT * (Estimation ratio * Time to the nearest vacant vehicle)
        GC_HV = Parameters.HV_const + \
                Parameters.HV_coef_fare * fare_HV + \
                Parameters.HV_coef_time * self.VoT * Parameters.phiHV * min(self.min_wait_time(HV_v), 1200)
        GC_AV = Parameters.AV_const + \
                Parameters.AV_coef_fare * fare_AV + \
                Parameters.AV_coef_time * self.VoT * Parameters.phiAV * min(self.min_wait_time(AV_v), 1200)

        # Logit choice based on GC (dis-utility) of vehicles
        _c = np.random.choice(['HV', 'AV', 'others'], p=np.exp([-GC_HV, -GC_AV, -Parameters.others_GC]) / sum(np.exp([-GC_HV, -GC_AV, -Parameters.others_GC])))
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

            # Record statistics
            Statistics.expiration_data = Statistics.expiration_data.append({'p_id': self.id, 'expire_t': self.expiredTime}, ignore_index=True)


class UpdatePhi(Event):
    def __init__(self, time):
        super().__init__(time, priority=2)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'UpdatePhi@t{}'.format(self.time)

    def trigger(self, nHV, nAV):
        Parameters.phiHV = compute_phi(len(Passenger.p_HV), nHV)
        Parameters.phiAV = compute_phi(len(Passenger.p_AV), nAV)


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
