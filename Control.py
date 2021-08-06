import numpy as np
import pandas as pd
import do_mpc


class Statistics:
    # Simulation outputs
    vehicle_data = []
    passenger_data = []
    expiration_data = []
    assignment_data = []
    utilisation_data = []

    # Simulation states
    lastPassengerTime = 0
    simulationEndTime = 0

    HV_total = 0
    HV_trips = 0
    HV_no = 0
    AV_trips = 0
    AV_no = 0


class Variables:
    # Passenger fare variables
    HVf1 = 3.0  # Default HV flag fare, $3.0
    HVf2 = 0.8 / 60  # Default HV unit fare, $0.8 / min
    AVf1 = 6.0  # Default AV flag fare, $6.0
    AVf2 = 0.6 / 60  # Default HV unit fare, $0.6 / min

    # Fixed generalised cost, representing alternative mode choices. Assume $5 trip cost + $20 waiting cost
    others_GC = 25

    # Driver wage variables
    unitWage = 40 / 3600  # Income = Unit wage (per second) * Trip duration

    # Driver join/exit market variables
    HV_utilisation = 0.6  # Initial utilisation ratio for expected revenue estimation
    AV_utilisation = 0.0  # AV utilisation for performance evaluation

    # ETA estimation model (ratios)
    phiHV = 1.0  # Default ETA ratio, function of nHV and pHV
    phiAV = 1.0  # Default ETA ratio, function of nAV and pAV


def set_wage(wage=None):
    if wage is not None:
        Variables.unitWage = wage


def compute_phi(nP, nV):
    # TODO: Improve estimation model
    less = min(nP, nV)
    more = max(nP, nV)
    return max(1.0, np.exp(0.16979338 + 0.03466977 * less - 0.0140257 * more))


def write_results(path, number):
    pd.DataFrame(Statistics.vehicle_data, columns=['v_id', 'is_HV', 'neoclassical', 'income', 'time', 'activation']
                 ).to_csv('{}/sim{}_vehicle_data.csv'.format(path, number), index=False)

    pd.DataFrame(Statistics.passenger_data,
                 columns=['p_id', 'request_t', 'trip_d', 'trip_t', 'VoT', 'fare', 'prefer_HV']
                 ).to_csv('{}/sim{}_passenger_data.csv'.format(path, number), index=False)

    pd.DataFrame(Statistics.expiration_data, columns=['p_id', 'expire_t']
                 ).to_csv('{}/sim{}_expiration_data.csv'.format(path, number), index=False)

    pd.DataFrame(Statistics.assignment_data, columns=['v_id', 'p_id', 'dispatch_t', 'meeting_t', 'delivery_t']
                 ).to_csv('{}/sim{}_assignment_data.csv'.format(path, number), index=False)

    pd.DataFrame(Statistics.utilisation_data, columns=['time', 'v_id', 'trip_utilisation']
                 ).to_csv('{}/sim{}_utilisation_data.csv'.format(path, number), index=False)
