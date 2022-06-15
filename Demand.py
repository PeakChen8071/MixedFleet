import pandas as pd
import numpy as np
from scipy.stats import truncnorm

from Parser import passenger_file, default_waiting_time
from Basics import Event, Location, duration_between, distance_between, compute_phi, p_id
from Control import Statistics, Parameters, Variables

passengers = {}


# File is validated to include passenger attributes for future simulations.
# Similar to using a random seed which maintains stochastic attributes over difference simulations.
def validate_passengers(file=passenger_file):
    if any(~pd.Series(['patience', 'trip_distance', 'trip_duration', 'U_const', 'U_fare', 'VoT']).isin(pd.read_csv(file, nrows=0))):
        print('Injecting passenger attributes...')
        df = pd.read_csv(file)

        df['tpep_pickup_datetime'] = (pd.to_datetime(df['tpep_pickup_datetime']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

        # Random patience time (sec) ~ Normal(60, 6^2) bounded by [30, 90]
        df['patience'] = truncnorm.rvs(a=-5, b=5, loc=60, scale=6, size=df.shape[0]).astype(int)

        # Calculate trip properties for access in future simulations
        df['trip_distance'] = df.apply(lambda x: distance_between(Location(x['o_source'], x['o_target'], x['o_loc']),
                                                                  Location(x['d_source'], x['d_target'], x['d_loc'])), axis=1)
        df['trip_duration'] = df.apply(lambda x: duration_between(Location(x['o_source'], x['o_target'], x['o_loc']),
                                                                  Location(x['d_source'], x['d_target'], x['d_loc'])), axis=1)

        # Passenger choice parameters for EV mode utility
        df['U_const'] = truncnorm.rvs(a=-1, b=1, loc=0, scale=1, size=df.shape[0])
        df['U_fare'] = truncnorm.rvs(a=-1, b=1, loc=3.2, scale=0.2, size=df.shape[0])

        # Random VoT ($/hr) ~ Normal(32, 3.2^2) bounded by [22, 38], rounded to int. (NYC HDM, 2018 household income)
        # VoT might be underestimated for Manhattan which is a relatively high-income area (Ulak, et al., 2020)
        df['VoT'] = truncnorm.rvs(a=-3.125, b=1.875, loc=32, scale=3.2, size=df.shape[0])
        df['VoT'] = df['VoT'].round(2)  # Round to the nearest cents for readability

        # Write back to passenger file with injected attributes
        df.sort_values('tpep_pickup_datetime').to_csv(file, index=False)
        print('Passenger attributes are successfully injected.')


def load_passengers(fraction=1, hours=18):
    use_cols = ['tpep_pickup_datetime', 'patience',
                'o_source', 'o_target', 'o_loc',
                'd_source', 'd_target', 'd_loc',
                'trip_distance', 'trip_duration',
                'U_const', 'U_fare', 'VoT']
    passenger_df = pd.read_csv(passenger_file, usecols=use_cols)
    passenger_df['time'] = passenger_df['tpep_pickup_datetime'] - passenger_df['tpep_pickup_datetime'].min()

    # Demand time shift, move 00:00 - 04:00 to the end of the day such that simulation starts at 04:00
    passenger_df['time'] = (passenger_df['time'] + 20 * 3600) % (24 * 3600)

    # Limit daily demand to the specified hours
    passenger_df = passenger_df[passenger_df['time'] <= hours * 3600]

    # print(passenger_df.dtypes)  # Print data types for debugging

    passenger_df = passenger_df.drop(columns=['tpep_pickup_datetime']).sample(frac=fraction).reset_index()

    Statistics.lastPassengerTime = passenger_df['time'].max()

    # Create passenger events
    for p in passenger_df.itertuples():
        NewPassenger(p.time, p.o_source, p.o_target, p.o_loc, p.d_source, p.d_target, p.d_loc,
                     p.trip_distance, p.trip_duration, p.patience, p.U_const, p.U_fare, p.VoT)


class Passenger:

    def __init__(self, time, origin, destination, trip_distance, trip_duration, patience, U_const, U_fare, VoT, EVs):
        self.id = next(p_id)
        self.requestTime = time
        self.origin = origin
        self.destination = destination
        self.tripDistance = trip_distance
        self.tripDuration = trip_duration
        self.expiredTime = time + patience

        self.U_const = U_const
        self.U_fare = U_fare / 60  # fare utility coefficient (1/sec)
        self.VoT = VoT / 3600  # Value of time ($/sec)
        self.fare = round(Parameters.baseFare + Parameters.unitFare * self.tripDuration / 3600, 2)

        generalised_cost = self.U_const + self.U_fare * self.fare + self.VoT * self.min_wait_time(EVs) * Parameters.phi
        self.preferEV = np.exp(-generalised_cost) / sum(np.exp([-generalised_cost, -Parameters.others_GC])) > Parameters.choices[self.id]

        if self.preferEV:  # Passenger waits for available EVs
            passengers[self.id] = self

        # Record statistics
        Statistics.data_output['passenger_data'].append([self.id, self.requestTime, self.tripDistance,
                                                         self.tripDuration, self.VoT, self.fare, self.preferEV])

    def __repr__(self):
        return 'Passenger{}'.format(self.id)

    def min_wait_time(self, vehicles):
        nearest_time = float('inf')
        for vehicle in vehicles:
            nearest_time = min(duration_between(vehicle.loc, self.origin), nearest_time)

        if nearest_time == float('inf'):
            return default_waiting_time
        else:
            return nearest_time

    def check_expiration(self, t):
        if t >= self.expiredTime:
            del passengers[self.id]
            Variables.EV_pw = len(passengers)

            # Record statistics
            Statistics.data_output['expiration_data'].append([self.id, self.expiredTime])


class NewPassenger(Event):
    def __init__(self, time, o_source, o_target, o_loc, d_source, d_target, d_loc, *args):
        super().__init__(time, priority=3)
        self.origin = Location(o_source, o_target, o_loc)
        self.destination = Location(d_source, d_target, d_loc)
        self.args = args

    def __repr__(self):
        return 'Passenger@t{}'.format(self.time)

    def trigger(self, EVs):
        Parameters.phi = compute_phi(len(passengers), len(EVs))  # Update waiting time estimation coefficient
        Passenger(self.time, self.origin, self.destination, *self.args, EVs)
        Variables.EV_pw = len(passengers)
