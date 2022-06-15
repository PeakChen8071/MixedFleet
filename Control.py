import csv
import numpy as np

from Parser import output_path, output_number


def write_results(folder=output_path, number=output_number):
    for file, data in Statistics.data_output.items():
        with open('{}/sim{}_{}.csv'.format(folder, number, file), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)


class Statistics:
    # Simulation outputs
    data_output = {'vehicle_data': [['v_id',  'time',  'neoclassical', 'hourly_cost', 'income_target', 'income', 'activation']],
                   'passenger_data': [['p_id', 'request_t', 'trip_d', 'trip_t', 'VoT', 'fare', 'prefer_EV']],
                   'expiration_data': [['p_id', 'expire_t']],
                   'assignment_data': [['v_id', 'p_id', 'is_cs', 'dispatch_t', 'meeting_t', 'delivery_t', 'profit', 'SoC']],
                   'utilisation_data': [['time', 'v_id', 'trip_utilisation']]}

    # Simulation time markers
    lastPassengerTime = 0
    simulationEndTime = 0


class Parameters:
    # Passenger Logit choice model
    choices = np.random.rand(int(1e6))
    # mean_const = 0
    # mean_U_fare = 3.2 / 60
    # mean_VoT = 32 / 3600
    others_GC = 50  # utility of alternative mode choices

    # Driver wage
    wage = 20  # Income = Unit wage (per hour) / 3600 * Trip duration (sec)
    # Passenger fare
    baseFare = 2.5
    unitFare = 48  # Default EV unit fare, $0.8 / min = $48 / hr

    # Order cancellation penalties
    psi = 4  # Penalty parameters for trip cancellation ($/trip)
    phi = 1.0  # Default ETA ratio, function of vacant vehicles and waiting passengers
    theta = 1.0  # Order cancellation rate estimation (pax/min)


class Variables:
    # System states
    total_trips = 0
    active_EVs = 0
    EV_pw = 0
    EV_nv = 0
    EV_na = 0
    EV_no = 0
    EV_nc = 0

    # Dynamic market condition indicators
    utilisation = 0.6
    occupancy = 0.6  # NOTE: The current version uses occupancy for supply decisions

