import math
import pandas as pd


vehicle_data = []
passenger_data = []
expiration_data = []
assignment_data = []
utilisation_data = []


class Variables:
    # Passenger fare variables
    HVf1 = 2.0  # Default HV flag fare, $2.0
    HVf2 = 2.0 / 1000  # Default HV unit fare, $2.0 / km
    AVf1 = 6.0  # Default AV flag fare, $6.0
    AVf2 = 1.5 / 1000  # Default AV unit fare, $1.5 / km

    # Fixed generalised cost, representing alternative mode choices. Assume $2 trip cost + $18 waiting cost
    others_GC = 20

    # Driver wage variables
    unitWage = 40 / 3600  # Income = Unit wage (per second) * Trip duration

    # Driver join/exit market variables
    HV_utilisation = 0.6  # Initial utilisation ratio for expected revenue estimation
    AV_utilisation = 0.0  # AV utilisation for performance evaluation

    # ETA estimation model (ratios)
    phiHV = 1.0  # Default ETA ratio, function of nHV and pHV
    phiAV = 1.0  # Default ETA ratio, function of nAV and pAV

    # Statistics
    HV_trips = 0
    AV_trips = 0


def compute_phi(nP, nV):
    # TODO: Improve estimation model
    less = min(nP, nV)
    more = max(nP, nV)
    return max(1.0, math.exp(0.16979338 + 0.03466977 * less - 0.0140257 * more))


def write_results(path, number):
    pd.DataFrame(vehicle_data, columns=['v_id', 'is_HV', 'income', 'time', 'activation']
                 ).to_csv('{}/sim{}_vehicle_data.csv'.format(path, number), index=False)

    pd.DataFrame(passenger_data, columns=['p_id', 'request_t', 'trip_d', 'trip_t', 'VoT', 'fare', 'prefer_HV']
                 ).to_csv('{}/sim{}_passenger_data.csv'.format(path, number), index=False)

    pd.DataFrame(expiration_data, columns=['p_id', 'expire_t']
                 ).to_csv('{}/sim{}_expiration_data.csv'.format(path, number), index=False)

    pd.DataFrame(assignment_data, columns=['v_id', 'p_id', 'dispatch_t', 'meeting_t', 'delivery_t']
                 ).to_csv('{}/sim{}_assignment_data.csv'.format(path, number), index=False)

    pd.DataFrame(utilisation_data, columns=['delivery_t', 'v_id', 'trip_utilisation']
                 ).to_csv('{}/sim{}_utilisation_data.csv'.format(path, number), index=False)
