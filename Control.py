import math
import pandas as pd


vehicle_data = {'deploy_t':{}, 'is_HV': {}}  # TODO: record trip OD (vehicle state transition)?
passenger_data = {'start_t': {}, 'expire_t': {}, 'trip_d': {}, 'fare': {}, 'prefer_HV': {}, 'expired': {}}
assignment_data = {'v_id': {}, 'p_id': {}, 'assignment_t': {}, 'dispatch_t': {}, 'delivery_t': {},
                   'dispatch_d': {}}


class Variables:
    # In general, flag price AVf1 > HVf1 and unit price AVf2 < HVf2
    HVf1 = 2.0  # Default HV flag fare, $2.0
    HVf2 = 2.0 / 1000  # Default HV unit fare, $2.0 / km
    AVf1 = 6.0  # Default AV flag fare, $6.0
    AVf2 = 1.5 / 1000  # Default AV unit fare, $1.5 / km

    unitWage = 20 / 100
    occupancy = 0.0
    # Join_Leave = U(unitWage, (average) occupancy)

    phiHV = 1.0  # Default approximation ratio, function of nHV and pHV
    phiAV = 1.0  # Default approximation ratio, function of nAV and pAV


def compute_phi(nP, nV):
    less = min(nP, nV)
    more = max(nP, nV)
    return max(1.0, math.exp(0.16979338 + 0.03466977 * less - 0.0140257 * more))


def write_results(path, number):
    pd.DataFrame.from_dict(vehicle_data).to_csv('{}/vehicle_data_{}.csv'
                                                .format(path, number), index_label='v_id')
    pd.DataFrame.from_dict(passenger_data).to_csv('{}/passenger_data_{}.csv'
                                                  .format(path, number), index_label='p_id')
    pd.DataFrame.from_dict(assignment_data).to_csv('{}/assignment_data_{}.csv'
                                                   .format(path, number), index_label='trip_id')
