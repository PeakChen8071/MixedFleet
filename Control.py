import numpy as np
import pandas as pd
import pyomo.environ as pyo

from Configuration import configs
from Basics import Event, population_start_time


def round_int(num, base):
    return base * round(num // base)


def compute_phi(nP, nV):
    less = min(nP, nV)
    more = max(nP, nV)
    phi = 1  # Default value

    if (less > 0) and (more > 0):
        phi = np.exp(0.185472) * less ** 0.199586 * more ** (-0.122311)

    return max(1.0, phi)


def write_results(path, number):
    pd.DataFrame(Statistics.vehicle_data).to_csv('{}/sim{}_vehicle_data.csv'.format(path, number),
                                                 header=['v_id', 'is_HV', 'neoclassical', 'hourly_cost',
                                                         'target_income', 'income', 'time', 'activation'],
                                                 index=False)
    pd.DataFrame(Statistics.passenger_data).to_csv('{}/sim{}_passenger_data.csv'.format(path, number),
                                                   header=['p_id', 'request_t', 'trip_d', 'trip_t', 'VoT', 'fare', 'prefer_HV'],
                                                   index=False)
    pd.DataFrame(Statistics.expiration_data).to_csv('{}/sim{}_expiration_data.csv'.format(path, number),
                                                    header=['p_id', 'expire_t'],
                                                    index=False)
    pd.DataFrame(Statistics.assignment_data).to_csv('{}/sim{}_assignment_data.csv'.format(path, number),
                                                    header=['v_id', 'p_id', 'is_HV', 'dispatch_t', 'meeting_t', 'delivery_t'],
                                                    index=False)
    pd.DataFrame(Statistics.utilisation_data).to_csv('{}/sim{}_utilisation_data.csv'.format(path, number),
                                                     header=['time', 'v_id', 'trip_utilisation'],
                                                     index=False)
    pd.DataFrame(Statistics.prediction_data).to_csv('{}/sim{}_prediction_data.csv'.format(path, number),
                                                    header=['time', 'objective', 'total_demand', 'AV_pw', 'AV_nv',
                                                            'AV_na', 'AV_no', 'AV_match', 'AV_exp', 'AV_pickup_param',
                                                            'AV_pickup', 'AV_dropoff_param', 'AV_dropoff', 'AV_U',
                                                            'HV_pw', 'HV_nv', 'HV_na', 'HV_no', 'HV_match', 'HV_exp',
                                                            'HV_pickup_param', 'HV_pickup', 'HV_dropoff_param',
                                                            'HV_dropoff', 'HV_supply', 'HV_exit_ratio', 'HV_U'],
                                                    index=False)
    pd.DataFrame(Statistics.control_data).to_csv('{}/sim{}_control_data.csv'.format(path, number),
                                                 header=['time', 'AV_unitFare', 'HV_unitFare', 'AV_supply'],
                                                 index=False)


class Statistics:
    # Simulation outputs
    vehicle_data = []
    passenger_data = []
    expiration_data = []
    assignment_data = []
    utilisation_data = []
    prediction_data = []
    control_data = []

    # Simulation time markers
    lastPassengerTime = 0
    simulationEndTime = 0

    # MPC setup parameters
    tau_c = configs['MPC_control_interval']
    tau_k = configs['MPC_prediction_interval']

    # Historical data & Exogenous demand and supply inputs
    histDemand = pd.read_csv(configs['historical_demand_file']).squeeze().to_list()
    histSupply, _ = np.histogram(population_start_time, bins=np.arange(0, 64800, configs['MPC_prediction_interval']))
    histSupply = list(histSupply * np.random.normal(1, 0.1, len(histSupply)))

    historicalTrips = pd.read_csv(configs['historical_trip_file'], index_col=0)
    historical_AV_trips = historicalTrips[~historicalTrips['is_HV']]
    historical_HV_trips = historicalTrips[historicalTrips['is_HV']]

    AV_pickup_counter = dict()
    AV_dropoff_counter = dict()
    HV_pickup_counter = dict()
    HV_dropoff_counter = dict()

    AV_pickup_time = historical_AV_trips['meeting_t'] - historical_AV_trips['dispatch_t']
    AV_pickup_hist = np.histogram(AV_pickup_time, bins=np.arange(0, AV_pickup_time.max(), tau_k), density=True)
    AV_dropoff_time = historical_AV_trips['delivery_t'] - historical_AV_trips['meeting_t']
    AV_dropoff_hist = np.histogram(AV_dropoff_time, bins=np.arange(0, AV_dropoff_time.max(), tau_k), density=True)

    HV_pickup_time = historical_HV_trips['meeting_t'] - historical_HV_trips['dispatch_t']
    HV_pickup_hist = np.histogram(HV_pickup_time, bins=np.arange(0, HV_pickup_time.max(), tau_k), density=True)
    HV_dropoff_time = historical_HV_trips['delivery_t'] - historical_HV_trips['meeting_t']
    HV_dropoff_hist = np.histogram(HV_dropoff_time, bins=np.arange(0, HV_dropoff_time.max(), tau_k), density=True)

    AV_new_pickup = list(np.random.choice(AV_pickup_hist[1][1:], int(1e7), p=tau_k * AV_pickup_hist[0]))
    AV_new_dropoff = list(np.random.choice(AV_dropoff_hist[1][1:], int(1e7), p=tau_k * AV_dropoff_hist[0]))
    HV_new_pickup = list(np.random.choice(HV_pickup_hist[1][1:], int(1e7), p=tau_k * HV_pickup_hist[0]))
    HV_new_dropoff = list(np.random.choice(HV_dropoff_hist[1][1:], int(1e7), p=tau_k * HV_dropoff_hist[0]))

    del historical_AV_trips, historical_HV_trips, \
        AV_pickup_time, AV_dropoff_time, HV_pickup_time, HV_dropoff_time, \
        AV_pickup_hist, AV_dropoff_hist, HV_pickup_hist, HV_dropoff_hist


class Parameters:
    # Base fare, constant flag-off price
    AV_base_fare = 2.5
    HV_base_fare = 2.5

    # Passenger Logit choice model
    AV_const = 0
    HV_const = 0
    AV_coef_fare = 3.2
    HV_coef_fare = 3.2
    mean_VoT = 32 / 3600  # Assume passenger VoT is represented by the mean
    U_scale = 0.1  # Utility scale for Logit choice model
    others_GC = 6  # scaled utility of alternative mode choices

    # AV cost
    AV_vacant_cost = 0.001  # Operational cost per veh·sec
    AV_operational_cost = 0.002  # Operational cost per veh·sec

    # Order cancellation penalties
    AV_penalty = 5  # Penalty parameters for optimisation
    HV_penalty = 5  # $ per cancellation
    others_penalty = 1

    beta = 3.5517737906200484  # Order cancellation rate estimation
    decay = 0.8  # Market prospect smoothing (decay) factor

    # Time-varying parameters
    HV_utilisation = 0.6
    AV_utilisation = 0
    HV_occupancy = 0.6  # NOTE: The current version uses occupancy for supply decisions
    AV_occupancy = 0

    phiHV = 1.0  # Default ETA ratio, function of nHV and pHV
    phiAV = 1.0  # Default ETA ratio, function of nAV and pAV


class Variables:
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

    # Market condition default values
    HV_ta = 300
    HV_to = 600
    AV_ta = 300
    AV_to = 600

    # Fixed driver wage
    HV_wage = 30  # Income = Unit wage (per hour) / 3600 * Trip duration (sec)

    # Passenger fare control inputs
    AV_unitFare = 36  # Default AV unit fare, $0.6 / min = $36 / hr
    HV_unitFare = 36  # Default HV unit fare, $0.6 / min = $36 / hr

    # AV fleet size control input
    AV_change = 0

    totalWage = 0
    exitDecisions = 0


def pyoMin(a, b):
    return pyo.Expr_if(IF=a > b, THEN=b, ELSE=a)


def pyoMax(a, b):
    return pyo.Expr_if(IF=a > b, THEN=a, ELSE=b)


def mpc_mixed_fleet(N, Nc, tau_c, tau_k, HV_pickup_time):
    # Define a finite-horizon m, with N+1 steps, each lasts tau_c seconds
    m = pyo.ConcreteModel()
    horizon = int(N * tau_c)
    m.t = pyo.RangeSet(0, horizon - tau_k, tau_k)
    m.k = pyo.RangeSet(0, Nc - 1)
    m.states = pyo.Set(initialize=['AV_pw', 'AV_nv', 'AV_na', 'AV_no',
                                   'HV_pw', 'HV_nv', 'HV_na', 'HV_no'])

    # Fix initial states from simulator
    m.x = pyo.Var(m.states, m.t, within=pyo.NonNegativeReals)  # System state variables

    # Control variables
    m.AV_fare = pyo.Var(m.k, bounds=[0, 180])
    m.HV_fare = pyo.Var(m.k, bounds=[0, 180])
    m.AV_fleet = pyo.Var(m.k, initialize=0, bounds=[-Variables.AV_nv, configs['AV_fleet_size'] - Variables.AV_total])

    # Dynamic parameters
    m.totalDemand = pyo.Param(m.t, initialize=0, mutable=True)
    m.HV_supply = pyo.Param(m.t, initialize=0, mutable=True)

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
                new_t = est_t + round_int(HV_pickup_time, tau_k)
            elif est_state == 'HV_dropoff':
                new_t = est_t + round_int(HV_pickup_time, tau_k) + Statistics.HV_new_dropoff.pop()
            if new_t < horizon:
                correction_est[(est_state, new_t)].append(est_t)

    # Auxiliary expressions
    def AV_expiration(model, t):
        return Parameters.beta * tau_k / tau_c * pyoMax(model.x['AV_pw', t] - model.x['AV_nv', t], 0)

    def AV_match(model, t):
        return pyoMin(model.x['AV_pw', t], model.x['AV_nv', t])

    def HV_expiration(model, t):
        return Parameters.beta * tau_k / tau_c * pyoMax(model.x['HV_pw', t] - model.x['HV_nv', t], 0)

    def HV_match(model, t):
        return pyoMin(model.x['HV_pw', t], model.x['HV_nv', t])

    def HV_exit_ratio(model, t):
        neoclassical_exit = configs['neoclassical'] / (1 + pyo.exp(13.4 * (model.x['HV_no', t] / Variables.HV_total - 0.432)))
        income_exit = (1 - configs['neoclassical']) * (0.0104 * pyo.exp(0.0211 * ((Variables.totalWage + Variables.HV_wage * Variables.HV_no / 3600 * sum(model.HV_match[i] for i in range(0, t-tau_k, tau_k))) / Variables.HV_total)) - 0.0104)
        return income_exit + neoclassical_exit

    def new_correction(model, target, t):
        if target == 'AV_pickup' or 'AV_dropoff':
            return model.correction_param[target, t] + sum(model.AV_match[t_est] for t_est in correction_est[(target, t)])
        elif target == 'HV_pickup' or 'HV_dropoff':
            return model.correction_param[target, t] + sum(model.HV_match[t_est] for t_est in correction_est[(target, t)])

    m.AV_exp = pyo.Expression(m.t, expr=AV_expiration)
    m.AV_match = pyo.Expression(m.t, expr=AV_match)
    m.HV_exp = pyo.Expression(m.t, expr=HV_expiration)
    m.HV_match = pyo.Expression(m.t, expr=HV_match)

    m.correction = pyo.Expression(m.correction_idx, m.t, rule=new_correction)

    m.HV_exit_ratio = pyo.Expression(m.t, expr=HV_exit_ratio)

    # Passenger choice prediction
    m.AV_GC = pyo.Expression(m.t, rule=lambda m, t: Parameters.U_scale * (Parameters.AV_const + Parameters.AV_coef_fare *
                                                    (Parameters.AV_base_fare + m.AV_fare[int(t >= tau_c)] * 120 * np.log(Variables.AV_to) / 3600)
                                                    + Parameters.mean_VoT * Variables.AV_ta))
    m.HV_GC = pyo.Expression(m.t, rule=lambda m, t: Parameters.U_scale * (Parameters.HV_const + Parameters.HV_coef_fare *
                                                    (Parameters.HV_base_fare + m.HV_fare[int(t >= tau_c)] * 120 * np.log(Variables.HV_to) / 3600)
                                                    + Parameters.mean_VoT * Variables.HV_ta))
    m.AV_U = pyo.Expression(m.t, rule=lambda m, t: pyo.exp(-m.AV_GC[t]) / (pyo.exp(-m.AV_GC[t]) + pyo.exp(-m.HV_GC[t]) + pyo.exp(-Parameters.others_GC)))
    m.HV_U = pyo.Expression(m.t, rule=lambda m, t: pyo.exp(-m.HV_GC[t]) / (pyo.exp(-m.AV_GC[t]) + pyo.exp(-m.HV_GC[t]) + pyo.exp(-Parameters.others_GC)))

    def system_dynamics(model, s, t):
        if t == horizon - tau_k:
            return pyo.Constraint.Skip
        else:
            if s == 'AV_pw':
                return model.x[s, t + tau_k] == model.x[s, t] + model.totalDemand[t] * model.AV_U[t] - model.AV_match[t] - model.AV_exp[t]
            elif s == 'AV_nv':
                if t % tau_c == 0:
                    return model.x[s, t + tau_k] == model.x[s, t] + model.AV_fleet[int(t >= tau_c)] - model.AV_match[t] + model.correction['AV_dropoff', t]
                else:
                    return model.x[s, t + tau_k] == model.x[s, t] - model.AV_match[t] + model.correction['AV_dropoff', t]
            elif s == 'AV_na':
                return model.x[s, t + tau_k] == model.x[s, t] + model.AV_match[t] - model.correction['AV_pickup', t]
            elif s == 'AV_no':
                return model.x[s, t + tau_k] == model.x[s, t] + model.correction['AV_pickup', t] - model.correction['AV_dropoff', t]
            elif s == 'HV_pw':
                return model.x[s, t + tau_k] == model.x[s, t] + model.totalDemand[t] * model.HV_U[t] - model.HV_match[t] - model.HV_exp[t]
            elif s == 'HV_nv':
                return model.x[s, t + tau_k] == model.x[s, t] + model.HV_supply[t] - model.HV_match[t] + (1 - 0.5 * model.HV_exit_ratio[t]) * model.correction['HV_dropoff', t]
            elif s == 'HV_na':
                return model.x[s, t + tau_k] == model.x[s, t] + model.HV_match[t] - model.correction['HV_pickup', t]
            elif s == 'HV_no':
                return model.x[s, t + tau_k] == model.x[s, t] + model.correction['HV_pickup', t] - model.correction['HV_dropoff', t]

    m.state_dynamics = pyo.Constraint(m.states, m.t, rule=system_dynamics)

    # Objective function
    def max_profit(model):
        AV_gain = sum(model.AV_match[t] * (model.AV_fare[int(t >= tau_c)] * Variables.AV_to / 3600 + Parameters.AV_base_fare) for t in model.t)
        AV_cost = tau_k * sum((Parameters.AV_operational_cost * (model.x['AV_na', t] + model.x['AV_no', t])) +
                              Parameters.AV_vacant_cost * model.x['AV_nv', t] for t in model.t)
        HV_profit = sum(model.HV_match[t] * ((model.HV_fare[int(t >= tau_c)] - Variables.HV_wage) * Variables.HV_to / 3600 + Parameters.HV_base_fare) for t in model.t)
        AV_penalty = Parameters.AV_penalty * pyo.summation(model.AV_exp)
        HV_penalty = Parameters.HV_penalty * pyo.summation(model.HV_exp)
        others_penalty = Parameters.others_penalty * sum(model.totalDemand[t] * (1 - model.AV_U[t] - model.HV_U[t]) for t in model.t)
        return AV_gain - AV_cost + HV_profit - AV_penalty - HV_penalty - others_penalty

    m.obj = pyo.Objective(expr=max_profit, sense=pyo.maximize)
    return m


class MPC(Event):
    def __init__(self, time, N=6, Nc=2, tau_c=300, tau_k=10):
        super().__init__(time, priority=6)
        self.N = N  # Number of control intervals
        self.Nc = Nc  # Number of active control intervals (after which controls are unchanged)
        self.tau_c = tau_c  # Duration (in seconds) of a control interval
        self.tau_k = tau_k  # Duration of a prediction step, for [k = tau_c / tau_k] steps in each control interval

    def __repr__(self):
        return 'MPC@t{}'.format(self.time)

    def trigger(self):
        pickup_time_estimate = Variables.HV_ta
        if (Variables.HV_pw > 0) or (Variables.HV_nv > 0):
            pickup_time_estimate = np.exp(7.597474) * min(Variables.HV_pw, Variables.HV_nv) ** 0.189208 * max(Variables.HV_pw, Variables.HV_nv) ** (-0.579565)

        # Construct MPC model
        MPC_model = mpc_mixed_fleet(N=self.N, Nc=self.Nc, tau_c=self.tau_c, tau_k=self.tau_k, HV_pickup_time=pickup_time_estimate)

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
                MPC_model.correction_param['AV_pickup', round_int(k - self.time, self.tau_k)] += v

        for k, v in Statistics.HV_pickup_counter.copy().items():
            if k < self.time:
                Statistics.HV_pickup_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['HV_pickup', round_int(k - self.time, self.tau_k)] += v

        for k, v in Statistics.AV_dropoff_counter.copy().items():
            if k < self.time:
                Statistics.AV_dropoff_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['AV_dropoff', round_int(k - self.time, self.tau_k)] += v

        for k, v in Statistics.HV_dropoff_counter.copy().items():
            if k < self.time:
                Statistics.HV_dropoff_counter.pop(k)
            elif k < self.time + self.N * self.tau_c:
                MPC_model.correction_param['HV_dropoff', round_int(k - self.time, self.tau_k)] += v

        # Part 3: exogenous demand and supply
        for t in MPC_model.t:
            MPC_model.totalDemand[t] = Statistics.histDemand[int((self.time + t) / self.tau_k)]
            MPC_model.HV_supply[t] = Statistics.histSupply[int((self.time + t) / self.tau_k)]

        # Part 4: load default control values
        for k in MPC_model.k:
            MPC_model.AV_fare[k] = Variables.AV_unitFare
            MPC_model.HV_fare[k] = Variables.HV_unitFare

        # TODO: Fix/Unfix control values here
        # MPC_model.AV_fare.fix()
        # MPC_model.HV_fare.fix()
        # MPC_model.AV_fleet.fix()

        # Solve MPC optimisation
        solver = pyo.SolverFactory('ipopt')

        result = solver.solve(MPC_model, load_solutions=False)
        # result = solver.solve(MPC_model, load_solutions=False, tee=True)

        if str(result.solver.status) != 'error':
            MPC_model.solutions.load_from(result)  # Load optimisation results

            # Apply (the first/immediate) control inputs
            Variables.AV_unitFare = round(MPC_model.AV_fare[0](), 2)
            Variables.HV_unitFare = round(MPC_model.HV_fare[0](), 2)
            Variables.AV_change = round(MPC_model.AV_fleet[0]())

            # Record optimisation results
            Statistics.prediction_data.append([self.time, MPC_model.obj(), MPC_model.totalDemand[:](),
                                               MPC_model.x['AV_pw', :](), MPC_model.x['AV_nv', :](),
                                               MPC_model.x['AV_na', :](), MPC_model.x['AV_no', :](),
                                               MPC_model.AV_match[:](), MPC_model.AV_exp[:](),
                                               MPC_model.correction_param['AV_pickup', :](),
                                               MPC_model.correction['AV_pickup', :](),
                                               MPC_model.correction_param['AV_dropoff', :](),
                                               MPC_model.correction['AV_dropoff', :](),
                                               MPC_model.AV_U[:](), MPC_model.x['HV_pw', :](),
                                               MPC_model.x['HV_nv', :](), MPC_model.x['HV_na', :](),
                                               MPC_model.x['HV_no', :](), MPC_model.HV_match[:](),
                                               MPC_model.HV_exp[:](),
                                               MPC_model.correction_param['HV_pickup', :](),
                                               MPC_model.correction['HV_pickup', :](),
                                               MPC_model.correction_param['HV_dropoff', :](),
                                               MPC_model.correction['HV_dropoff', :](),
                                               MPC_model.HV_supply[:](),
                                               MPC_model.HV_exit_ratio[:](), MPC_model.HV_U[:]()])

            Statistics.control_data.append([self.time, MPC_model.AV_fare[:](), MPC_model.HV_fare[:](), MPC_model.AV_fleet[:]()])

            # Display optimisation results
            print('AV_unitFare = {}; HV_unitFare = {}\nAV_supply = {}\nObjective = {}'.format(
                MPC_model.AV_fare[:](),
                MPC_model.HV_fare[:](),
                MPC_model.AV_fleet[:](),
                MPC_model.obj()))

            print('AV_U : HV_U = {:.3f} : {:.3f}'.format(MPC_model.AV_U[0](), MPC_model.HV_U[0]()))

        else:  # Use default control values if optimisation fails
            print('Optimisation failed! Time = ', self.time)
            # Variables.AV_unitFare = 36
            # Variables.HV_unitFare = 36
            Variables.AV_change = 0
