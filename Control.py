import math
import pandas as pd


vehicle_data = []
passenger_data = []
expiration_data = []
assignment_data = []


class Variables:
    # In general, flag price AVf1 > HVf1 and unit price AVf2 < HVf2
    HVf1 = 2.0  # Default HV flag fare, $2.0
    HVf2 = 2.0 / 1000  # Default HV unit fare, $2.0 / km
    AVf1 = 6.0  # Default AV flag fare, $6.0
    AVf2 = 1.5 / 1000  # Default AV unit fare, $1.5 / km

    unitWage = 20 / 100
    occupancy = 0.0  # mean(Occupied time / Vacant time)
    # Join_Leave = U(unitWage, (average) occupancy)

    phiHV = 1.0  # Default approximation ratio, function of nHV and pHV
    phiAV = 1.0  # Default approximation ratio, function of nAV and pAV


def compute_phi(nP, nV):
    less = min(nP, nV)
    more = max(nP, nV)
    return max(1.0, math.exp(0.16979338 + 0.03466977 * less - 0.0140257 * more))


def write_results(path, number):
    pd.DataFrame(vehicle_data, columns=['v_id', 'is_HV', 'time', 'activation']
                 ).to_csv('{}/vehicle_data_{}.csv'.format(path, number), index=False)

    pd.DataFrame(passenger_data, columns=['p_id', 'start_t', 'trip_d', 'VoT', 'fare', 'prefer_HV']
                 ).to_csv('{}/passenger_data_{}.csv'.format(path, number), index=False)

    pd.DataFrame(expiration_data, columns=['p_id', 'expire_t']
                 ).to_csv('{}/expiration_data_{}.csv'.format(path, number), index=False)

    pd.DataFrame(assignment_data, columns=['v_id', 'p_id', 'dispatch_t', 'meeting_t', 'delivery_t', 'dispatch_d']
                 ).to_csv('{}/assignment_data_{}.csv'.format(path, number), index=False)
