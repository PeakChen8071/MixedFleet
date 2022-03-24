import numpy as np
import pandas as pd
import pyomo.environ as pyo

from Configuration import configs
from Basics import Event


def round_int(num, base):
    return base * round(num // base)


def read_historical_data(path, trip_file, vehicle_file):
    historicalTrips = pd.read_csv(path + trip_file)
    historicalVehicles = pd.read_csv(path + vehicle_file)
    historicalAVs = historicalVehicles[~historicalVehicles['is_HV']]['v_id'].tolist()
    historicalHVs = historicalVehicles[historicalVehicles['is_HV']]['v_id'].tolist()
    historicalAVTrips = historicalTrips[historicalTrips['v_id'].isin(historicalAVs)]
    historicalHVTrips = historicalTrips[historicalTrips['v_id'].isin(historicalHVs)]

    return historicalAVTrips, historicalHVTrips


def compute_phi(nP, nV):
    # TODO: Improve estimation model
    less = min(nP, nV)
    more = max(nP, nV)
    return max(1.0, np.exp(0.16979338 + 0.03466977 * less - 0.0140257 * more))


def write_results(path, number):
    Statistics.vehicle_data.to_csv('{}/sim{}_vehicle_data.csv'.format(path, number))
    Statistics.passenger_data.to_csv('{}/sim{}_passenger_data.csv'.format(path, number))
    Statistics.expiration_data.to_csv('{}/sim{}_expiration_data.csv'.format(path, number))
    Statistics.assignment_data.to_csv('{}/sim{}_assignment_data.csv'.format(path, number))
    Statistics.utilisation_data.to_csv('{}/sim{}_utilisation_data.csv'.format(path, number))
    Statistics.prediction_data.to_csv('{}/sim{}_prediction_data.csv'.format(path, number))
    Statistics.control_data.to_csv('{}/sim{}_control_data.csv'.format(path, number))
    Statistics.objective_data.to_csv('{}/sim{}_objective_data.csv'.format(path, number))

    pd.DataFrame(Variables.histExits).to_csv('{}/sim{}_exit_data.csv'.format(path, number))


class Statistics:
    tau_c = configs['MPC_control_interval']
    tau_k = configs['MPC_prediction_interval']

    # Simulation outputs
    vehicle_data = pd.DataFrame(columns=['v_id', 'is_HV', 'neoclassical', 'income', 'time', 'activation'])
    passenger_data = pd.DataFrame(columns=['p_id', 'request_t', 'trip_d', 'trip_t', 'VoT', 'fare', 'prefer_HV'])
    expiration_data = pd.DataFrame(columns=['p_id', 'expire_t'])
    assignment_data = pd.DataFrame(columns=['v_id', 'p_id', 'is_HV', 'dispatch_t', 'meeting_t', 'delivery_t'])
    utilisation_data = pd.DataFrame(columns=['time', 'v_id', 'trip_utilisation'])
    prediction_data = pd.DataFrame(columns=['time', 'AV_pw', 'AV_nv', 'AV_na', 'AV_no', 'AV_pickup', 'AV_dropoff',
                                            'HV_supply', 'HV_pw', 'HV_nv', 'HV_na', 'HV_no', 'HV_pickup', 'HV_dropoff'])
    control_data = pd.DataFrame(columns=['time', 'AV_unitFare', 'HV_unitFare', 'AV_supply'])
    objective_data = pd.DataFrame(columns=['time', 'objective'])

    # Simulation time markers
    lastPassengerTime = 0
    simulationEndTime = 0

    # Historical data
    historical_AV_trips, historical_HV_trips = read_historical_data(path='../Results/Simulation_Outputs/',
                                                                    trip_file='default_historical_trips.csv',
                                                                    vehicle_file='default_historical_vehicles.csv')

    AV_pickup_counter = dict()
    AV_dropoff_counter = dict()
    HV_pickup_counter = dict()
    HV_dropoff_counter = dict()

    AV_pickup_time = historical_AV_trips['meeting_t'] - historical_AV_trips['dispatch_t']
    AV_pickup_hist = np.histogram(AV_pickup_time, bins=[i for i in range(0, max(AV_pickup_time), tau_k)], density=True)
    AV_dropoff_time = historical_AV_trips['delivery_t'] - historical_AV_trips['meeting_t']
    AV_dropoff_hist = np.histogram(AV_dropoff_time, bins=[i for i in range(0, max(AV_dropoff_time), tau_k)], density=True)

    HV_pickup_time = historical_HV_trips['meeting_t'] - historical_HV_trips['dispatch_t']
    HV_pickup_hist = np.histogram(HV_pickup_time, bins=[i for i in range(0, max(HV_pickup_time), tau_k)], density=True)
    HV_dropoff_time = historical_HV_trips['delivery_t'] - historical_HV_trips['meeting_t']
    HV_dropoff_hist = np.histogram(HV_dropoff_time, bins=[i for i in range(0, max(HV_dropoff_time), tau_k)], density=True)

    AV_new_pickup = list(np.random.choice(AV_pickup_hist[1][1:], int(1e6), p=tau_k * AV_pickup_hist[0]))
    AV_new_dropoff = list(np.random.choice(AV_dropoff_hist[1][1:], int(1e6), p=tau_k * AV_dropoff_hist[0]))
    HV_new_pickup = list(np.random.choice(HV_pickup_hist[1][1:], int(1e6), p=tau_k * HV_pickup_hist[0]))
    HV_new_dropoff = list(np.random.choice(HV_dropoff_hist[1][1:], int(1e6), p=tau_k * HV_dropoff_hist[0]))

    del historical_AV_trips, historical_HV_trips, \
        AV_pickup_time, AV_dropoff_time, HV_pickup_time, HV_dropoff_time, \
        AV_pickup_hist, AV_dropoff_hist, HV_pickup_hist, HV_dropoff_hist


class Parameters:
    # Passenger Logit choice model
    AV_const = 0
    HV_const = 0
    AV_coef_fare = 0.2
    HV_coef_fare = 0.2
    AV_coef_time = 0.05
    HV_coef_time = 0.05
    mean_VoT = 32 / 3600  # Assume passenger VoT is represented by the mean
    others_GC = 3  # scaled utility of alternative mode choices

    # AV cost
    AV_vacant_cost = 0.001  # Operational cost per veh·sec
    AV_operational_cost = 0.002  # Operational cost per veh·sec

    # Order cancellation penalties
    AV_penalty = 1  # Penalty parameters for optimisation
    HV_penalty = 1  # $ per cancellation

    # Order cancellation rate estimation
    HV_beta = 4
    AV_beta = 4

    # Time-varying parameters
    HV_utilisation = 0.6
    AV_utilisation = 0
    HV_occupancy = 0.6  # NOTE: The current version uses occupancy for supply decisions
    AV_occupancy = 0

    phiHV = 1.0  # Default ETA ratio, function of nHV and pHV
    phiAV = 1.0  # Default ETA ratio, function of nAV and pAV


class Variables:
    # Exogenous demand and supply inputs
    histDemand = []
    histSupply = []
    histExits = []
    exitDecisions = 0

    # System states
    HV_trips = 0
    AV_trips = 0

    HV_total = 0
    HV_pw = 0
    HV_nv = 0
    HV_na = 0
    HV_no = 0
    AV_total = 0
    AV_pw = 0
    AV_nv = 0
    AV_na = 0
    AV_no = 0

    # Market condition prediction
    HV_ta = 420
    HV_to = 600
    AV_ta = 420
    AV_to = 600

    # Fixed driver wage
    HV_wage = 40  # Income = Unit wage (per hour) / 3600 * Trip duration (sec)

    # Passenger fare control inputs
    AV_unitFare = 48  # Default AV unit fare, $0.8 / min = $48 / hr
    HV_unitFare = 42  # Default HV unit fare, $0.7 / min = $42 / hr

    # AV fleet size control input
    AV_change = 0


def pyoMin(a, b):
    return pyo.Expr_if(IF=a > b, THEN=b, ELSE=a)


def pyoMax(a, b):
    return pyo.Expr_if(IF=a > b, THEN=a, ELSE=b)


def mpc_mixed_fleet(N, Nc, tau_c, tau_k):
    # Define a finite-horizon m, with N+1 steps, each lasts tau_c seconds
    m = pyo.ConcreteModel()
    horizon = N * tau_c
    m.t = pyo.RangeSet(0, horizon - tau_k, tau_k)
    m.k = pyo.RangeSet(0, Nc - 1)
    m.states = pyo.Set(initialize=['AV_pw', 'AV_nv', 'AV_na', 'AV_no',
                                   'HV_pw', 'HV_nv', 'HV_na', 'HV_no'])
    m.controls = pyo.Set(initialize=['AV_unitFare', 'HV_unitFare', 'AV_supply'])

    # Fix initial states from simulator
    m.x = pyo.Var(m.states, m.t, within=pyo.NonNegativeReals)  # System state variables

    # Control variable bounds
    def control_bounds(m, control, k):
        if control == 'AV_supply':
            return -Variables.AV_nv, configs['AV_fleet_size'] - Variables.AV_total
        elif control == 'AV_unitFare':
            return 0, 150
        elif control == 'HV_unitFare':
            return 0, 150

    # Control variables
    m.u = pyo.Var(m.controls, m.k, bounds=control_bounds)

    # Auxiliary expressions
    def AV_expiration(model, t):
        return Parameters.AV_beta * tau_k / tau_c * pyoMax(model.x['AV_pw', t] - model.x['AV_nv', t], 0)

    def AV_match(model, t):
        return pyoMin(model.x['AV_pw', t], model.x['AV_nv', t])

    def HV_expiration(model, t):
        return Parameters.HV_beta * tau_k / tau_c * pyoMax(model.x['HV_pw', t] - model.x['HV_nv', t], 0)

    def HV_match(model, t):
        return pyoMin(model.x['HV_pw', t], model.x['HV_nv', t])

    m.AV_exp = pyo.Expression(m.t, expr=AV_expiration)
    m.AV_match = pyo.Expression(m.t, expr=AV_match)
    m.HV_exp = pyo.Expression(m.t, expr=HV_expiration)
    m.HV_match = pyo.Expression(m.t, expr=HV_match)

    # Dynamic parameters
    m.totalDemand = pyo.Param(m.t, initialize=0, mutable=True)
    m.HV_supply = pyo.Param(m.t, initialize=0, mutable=True)

    m.AV_to = pyo.Param(initialize=Variables.AV_to)
    m.AV_ta = pyo.Param(initialize=Variables.AV_ta)
    m.HV_to = pyo.Param(initialize=Variables.HV_to)
    m.HV_ta = pyo.Param(initialize=Variables.HV_ta)

    m.correction_idx = pyo.Set(initialize=['AV_pickup', 'AV_dropoff', 'HV_pickup', 'HV_dropoff'])
    m.correction_param = pyo.Param(m.correction_idx, m.t, initialize=0, mutable=True)
    correction_est = {(s, t): [] for s in m.correction_idx for t in m.t}

    for est_state in m.correction_idx:
        for est_t in m.t:
            new_t = None
            if est_state == 'AV_pickup':
                new_t = est_t + Statistics.AV_new_pickup.pop()
            elif est_state == 'AV_dropoff':
                new_t = est_t + Statistics.AV_new_pickup.pop() + Statistics.AV_new_dropoff.pop()
            elif est_state == 'HV_pickup':
                new_t = est_t + Statistics.HV_new_pickup.pop()
            elif est_state == 'HV_dropoff':
                new_t = est_t + Statistics.HV_new_pickup.pop() + Statistics.HV_new_dropoff.pop()
            if new_t < horizon:
                correction_est[(est_state, new_t)].append(est_t)

    def new_correction(model, target, t):
        if target == 'AV_pickup' or 'AV_dropoff':
            return model.correction_param[target, t] + sum(model.AV_match[t_est] for t_est in correction_est[(target, t)])
        elif target == 'HV_pickup' or 'HV_dropoff':
            return model.correction_param[target, t] + sum(model.HV_match[t_est] for t_est in correction_est[(target, t)])

    m.correction = pyo.Expression(m.correction_idx, m.t, rule=new_correction)

    # Passenger choice prediction
    m.AV_GC = pyo.Expression(m.t, rule=lambda m, t: Parameters.AV_const
                                                    + Parameters.AV_coef_fare * m.u['AV_unitFare', int(t >= tau_c)] / 3600 * m.AV_to
                                                    + Parameters.AV_coef_time * Parameters.mean_VoT * m.AV_ta)
    m.HV_GC = pyo.Expression(m.t, rule=lambda m, t: Parameters.HV_const
                                                    + Parameters.HV_coef_fare * m.u['HV_unitFare', int(t >= tau_c)] / 3600 * m.HV_to
                                                    + Parameters.AV_coef_time * Parameters.mean_VoT * m.HV_ta)
    m.AV_U = pyo.Expression(m.t, rule=lambda m, t: pyo.exp(-m.AV_GC[t]) / (pyo.exp(-m.AV_GC[t]) + pyo.exp(-m.HV_GC[t]) + pyo.exp(-Parameters.others_GC)))
    m.HV_U = pyo.Expression(m.t, rule=lambda m, t: pyo.exp(-m.HV_GC[t]) / (pyo.exp(-m.AV_GC[t]) + pyo.exp(-m.HV_GC[t]) + pyo.exp(-Parameters.others_GC)))

    def system_dynamics(model, s, t):
        if t == horizon - tau_k:
            return pyo.Constraint.Skip
        else:
            _value = 0
            if s == 'AV_pw':
                _value = model.totalDemand[t] * model.AV_U[t] - model.AV_match[t] - model.AV_exp[t]
            elif s == 'AV_nv':
                if t % tau_c == 0:
                    _value = model.u['AV_supply', int(t >= tau_c)] - model.AV_match[t] + model.correction['AV_dropoff', t]
                else:
                    _value = model.correction['AV_dropoff', t] - model.AV_match[t]
            elif s == 'AV_na':
                _value = model.AV_match[t] - model.correction['AV_pickup', t]
            elif s == 'AV_no':
                _value = model.correction['AV_pickup', t] - model.correction['AV_dropoff', t]
            elif s == 'HV_pw':
                _value = model.totalDemand[t] * model.HV_U[t] - model.HV_match[t] - model.HV_exp[t]
            elif s == 'HV_nv':
                _value = model.HV_supply[t] - model.HV_match[t] + model.correction['HV_dropoff', t]
            elif s == 'HV_na':
                _value = model.HV_match[t] - model.correction['HV_pickup', t]
            elif s == 'HV_no':
                _value = model.correction['HV_pickup', t] - model.correction['HV_dropoff', t]
            return model.x[s, t + tau_k] == model.x[s, t] + _value

    m.state_dynamics = pyo.Constraint(m.states, m.t, rule=system_dynamics)

    # Objective function
    def max_profit(model):
        AV_gain = sum(model.AV_match[t] * model.u['AV_unitFare', int(t >= tau_c)] for t in model.t) * model.AV_to / 3600
        HV_gain = sum(model.HV_match[t] * model.u['HV_unitFare', int(t >= tau_c)] for t in model.t) * model.HV_to / 3600
        AV_cost = tau_k * sum((Parameters.AV_operational_cost * (model.x['AV_na', t] + model.x['AV_no', t])) +
                              Parameters.AV_vacant_cost * model.x['AV_nv', t] for t in model.t)
        HV_cost = pyo.summation(model.HV_match) * Variables.HV_wage * model.HV_to / 3600
        AV_penalty = Parameters.AV_penalty * pyo.summation(model.AV_exp)
        HV_penalty = Parameters.HV_penalty * pyo.summation(model.HV_exp)
        return AV_gain + HV_gain - AV_cost - HV_cost - AV_penalty - HV_penalty

    m.obj = pyo.Objective(expr=max_profit, sense=pyo.maximize)
    return m


class MPC(Event):
    def __init__(self, time, N=6, Nc=2, tau_c=300, tau_k=10):
        super().__init__(time, priority=5)  # TODO: check the priority of MPC optimisation
        self.N = N  # Number of control intervals
        self.Nc = Nc  # Number of active control intervals (after which controls are unchanged)
        self.tau_c = tau_c  # Duration (in seconds) of a control interval
        self.tau_k = tau_k  # Duration of a prediction step, for [k = tau_c / tau_k] steps in each control interval

    def __repr__(self):
        return 'MPC@t{}'.format(self.time)

    def trigger(self):
        # Construct MPC model
        MPC_model = mpc_mixed_fleet(N=self.N, Nc=self.Nc, tau_c=self.tau_c, tau_k=self.tau_k)

        # Configure MPC dynamic parameters (system feedback)
        # Part 1: Current simulation states
        MPC_model.x['AV_pw', 0].fix(Variables.AV_pw)
        MPC_model.x['AV_nv', 0].fix(Variables.AV_nv)
        MPC_model.x['AV_na', 0].fix(Variables.AV_na)
        MPC_model.x['AV_no', 0].fix(Variables.AV_no)
        MPC_model.x['HV_pw', 0].fix(Variables.HV_pw)
        MPC_model.x['HV_nv', 0].fix(Variables.HV_nv)
        MPC_model.x['HV_na', 0].fix(Variables.HV_na)
        MPC_model.x['HV_no', 0].fix(Variables.HV_no)

        # Part 2: pick-up / drop-off correction parameters
        for k, v in Statistics.AV_pickup_counter.copy().items():
            if k < self.time:
                Statistics.AV_pickup_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['AV_pickup', round_int(k - self.time, self.tau_k)] = v

        for k, v in Statistics.HV_pickup_counter.copy().items():
            if k < self.time:
                Statistics.HV_pickup_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['HV_pickup', round_int(k - self.time, self.tau_k)] = v

        for k, v in Statistics.AV_dropoff_counter.copy().items():
            if k < self.time:
                Statistics.AV_dropoff_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['AV_dropoff', round_int(k - self.time, self.tau_k)] = v

        for k, v in Statistics.HV_dropoff_counter.copy().items():
            if k < self.time:
                Statistics.HV_dropoff_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['HV_dropoff', round_int(k - self.time, self.tau_k)] = v

        # Part 3: exogenous demand and supply
        for t in MPC_model.t:
            MPC_model.totalDemand[t] = Variables.histDemand[int((self.time + t) / self.tau_k)]
            MPC_model.HV_supply[t] = Parameters.HV_occupancy * Variables.histSupply[int((self.time + t) / self.tau_k)]

        # Part 4: load default control values
        for k in MPC_model.k:
            MPC_model.u['AV_unitFare', k] = Variables.AV_unitFare
            MPC_model.u['HV_unitFare', k] = Variables.HV_unitFare
            MPC_model.u['AV_supply', k] = 0

        # MPC_model.u.fix()  # TODO: Fix/Unfix control values here

        # Solve MPC optimisation
        solver = pyo.SolverFactory('ipopt')
        # solver.options['tol'] = 5
        # solver.options['max_iter'] = 10000
        solver.options['OF_tiny_step_tol'] = 1e-3
        # solver.options['OF_acceptable_obj_change_tol'] = 50
        solver.options['OF_tiny_step_y_tol'] = 1
        # solver.options['constr_viol_tol'] = 0.1
        # solver.options['OF_gamma_theta'] = 1e-2

        result = solver.solve(MPC_model, load_solutions=False)
        # result = solver.solve(MPC_model, load_solutions=False, tee=True)

        if str(result.solver.status) != 'error':
            MPC_model.solutions.load_from(result)

            Statistics.prediction_data = Statistics.prediction_data.append({'time': self.time,
                                                                            'AV_pw': MPC_model.x['AV_pw', :](),
                                                                            'AV_nv': MPC_model.x['AV_nv', :](),
                                                                            'AV_na': MPC_model.x['AV_na', :](),
                                                                            'AV_no': MPC_model.x['AV_no', :](),
                                                                            'AV_pickup': MPC_model.correction['AV_pickup', :](),
                                                                            'AV_dropoff': MPC_model.correction['AV_dropoff', :](),
                                                                            'HV_supply': MPC_model.HV_supply[:](),
                                                                            'HV_pw': MPC_model.x['HV_pw', :](),
                                                                            'HV_nv': MPC_model.x['HV_nv', :](),
                                                                            'HV_na': MPC_model.x['HV_na', :](),
                                                                            'HV_no': MPC_model.x['HV_no', :](),
                                                                            'HV_pickup': MPC_model.correction['HV_pickup', :](),
                                                                            'HV_dropoff': MPC_model.correction['HV_dropoff', :]()},
                                                                           ignore_index=True)

            Statistics.control_data = Statistics.control_data.append({'time': self.time,
                                                                      'AV_unitFare': MPC_model.u['AV_unitFare', :](),
                                                                      'HV_unitFare': MPC_model.u['HV_unitFare', :](),
                                                                      'AV_supply': MPC_model.u['AV_supply', :]()},
                                                                     ignore_index=True)

            Statistics.objective_data = Statistics.objective_data.append({'time': self.time,
                                                                          'objective': MPC_model.obj()},
                                                                         ignore_index=True)

            print('Time = {}\nAV_unitFare = {}\nHV_unitFare = {}\nAV_supply = {}'.format(
                self.time,
                MPC_model.u['AV_unitFare', :](),
                MPC_model.u['HV_unitFare', :](),
                MPC_model.u['AV_supply', :]()))

            print('Objective =', MPC_model.obj())

            print('AV:HV = {:.2f}:{:.2f}'.format(np.mean(MPC_model.AV_U[:]()), np.mean(MPC_model.HV_U[:]())))
            # print('AV pw =', sum(MPC_model.x['AV_pw' ,:]()))
            # print('AV nv =', sum(MPC_model.x['AV_nv' ,:]()))

            # Apply (the first/immediate) control inputs
            Variables.AV_unitFare = round(MPC_model.u['AV_unitFare', 0](), 2)
            Variables.HV_unitFare = round(MPC_model.u['HV_unitFare', 0](), 2)
            Variables.AV_change = round(MPC_model.u['AV_supply', 0]())
        else:
            print('Optimisation failed! Time = ', self.time)
            Variables.AV_unitFare = 48
            Variables.HV_unitFare = 42
            Variables.AV_change = 0
