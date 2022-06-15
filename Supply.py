import numpy as np
from scipy.stats import truncnorm

from Parser import maximum_work, fleet_size
from Basics import Event, Location, random_loc, duration_between, v_id
from Energy import Electricity
from Control import Parameters, Variables, Statistics


EVs = {}


# def load_vehicles(neoclassical=0.5):
#     # Instantiate the total AV fleet as inactive at depots (chosen randomly)
#     for d in np.random.choice(depots, configs['AV_fleet_size']):
#         AV(0, Location(d))
#
#     # Activate random AVs as the initial fleet
#     ActivateAVs(0, configs['AV_initial_size'])
#
#     # Load HVs (neoclassical or income-targeting) with preferred shift start time and duration
#     total = configs['HV_fleet_size']
#     morning = int(0.35 * total)
#     afternoon = int(0.3 * total)
#     evening = total - morning - afternoon
#
#     # Preferred start times, note that they might be shifted by some demand pattern adjustments (-4)
#     shift_start = []
#     shift_start += [int(3600 * i) for i in truncnorm.rvs(-3, 3, 3, 1, morning)]
#     shift_start += [int(3600 * i) for i in truncnorm.rvs(-2, 2, 9, 2, afternoon)]
#     shift_start += [int(3600 * i) for i in truncnorm.rvs(-1, 1.5, 15, 2, evening)]
#     # np.random.shuffle(shift_start)
#
#     # Set HV supply prediction values
#     bins = [i for i in range(0, Statistics.lastPassengerTime, configs['MPC_prediction_interval'])]
#     Variables.histSupply, _ = np.histogram(shift_start, bins=bins)
#
#     neoList = [k <= neoclassical for k in np.random.rand(total)]  # Proportion of neoclassical HVs
#
#     hourlyCost = list(np.random.uniform(10, 40, total))
#     # hourlyCost = list(truncnorm.rvs(a=-0.5, b=3, loc=18, scale=10, size=total))
#     targetIncome = list(np.random.uniform(50, 300, total))
#
#     for i in range(total):
#         NewHV(shift_start[i], random_loc(), neoList[i], hourlyCost[i], targetIncome[i])
#
#     # TODO: remove after supply test
#     # pd.DataFrame({'start_times': shift_start, 'neoclassical': neoList}).to_csv('../Results/Simulation_Outputs/test_supply_data.csv')


def load_simple_vehicles():
    shift_start = [int(i) for i in np.linspace(0, 3600, fleet_size)]
    SoC_level = np.random.uniform(0.9, 1.0, fleet_size)

    # Activate all EV fleet as
    for i in range(fleet_size):
        NewEV(shift_start[i], random_loc(), False, 0, 10000, SoC_level[i])


class NewEV(Event):
    def __init__(self, time, loc, neo, hourly_cost, target_income, SoC_level):
        super().__init__(time, priority=2)
        self.loc = loc
        self.neo = neo
        self.hourlyCost = hourly_cost
        self.targetIncome = target_income
        self.initialSoC = SoC_level * Electricity.max_SoC

    def __repr__(self):
        return 'NewEV@t{}'.format(self.time)

    def trigger(self):
        expected_wage = Parameters.wage * Variables.occupancy

        if ~self.neo or (expected_wage >= self.hourlyCost):
            EV(self.time, self.loc, self.neo, self.hourlyCost, self.targetIncome, self.initialSoC)
        elif self.neo and (self.time + 600 < Statistics.lastPassengerTime) and \
                ((self.hourlyCost - expected_wage) / self.hourlyCost < np.random.rand() - 0.2):
            # Neoclassical drivers may try to join the market again in 5 minutes (before last passenger)
            NewEV(self.time + 300, self.loc, self.neo, self.hourlyCost, self.targetIncome, self.initialSoC)


class EV:
    def __init__(self, time, loc, neo, hourly_cost, target_income, SoC):
        self.id = next(v_id)
        self.loc = loc

        self.entranceTime = time
        self.assignedTime = 0
        self.occupiedTime = 0

        self.neoclassical = neo
        self.hourlyCost = hourly_cost
        self.targetIncome = target_income
        self.income = 0
        self.SoC = SoC

        EVs[self.id] = self
        Variables.active_EVs += 1  # Update statistics
        Variables.EV_nv += 1

        self.nextTrip = None  # TripCompletion object, checked at planned destination
        self.destination = None  # Planned cruise destination, updated if intercepted by trip assignment

        # Record statistics
        Statistics.data_output['vehicle_data'].append([self.id, self.entranceTime, self.neoclassical, self.hourlyCost,
                                                       self.targetIncome, self.income, True])

    def __repr__(self):
        return 'EV{}'.format(self.id)

    def decide_exit(self, exit_time, end=False):
        EVs.pop(self.id, None)
        Variables.active_EVs -= 1

        if exit_time - self.entranceTime >= maximum_work or end:
            # Force exit labour market and record statistics
            Statistics.data_output['vehicle_data'].append([self.id, exit_time,  self.neoclassical, self.hourlyCost,
                                                           self.targetIncome, self.income, False])
        else:
            if self.neoclassical and ((self.hourlyCost - Parameters.wage * Variables.occupancy) / self.hourlyCost < np.random.rand() - 0.2):
                # Neoclassical drivers have a chance to continue working based on the expected wage
                EVs[self.id] = self
                Variables.active_EVs += 1
                Variables.EV_nv += 1

            elif ~self.neoclassical and (self.income < self.targetIncome):
                # Income-targeting drivers continue to work if accumulated income < target income
                EVs[self.id] = self
                Variables.active_EVs += 1
                Variables.EV_nv += 1
            else:
                # Exit labour market and record statistics
                Statistics.data_output['vehicle_data'].append([self.id, exit_time, self.neoclassical, self.hourlyCost,
                                                               self.targetIncome, self.income, False])


class TripCompletion(Event):
    def __init__(self, time, vehicle, drop_off=False):
        super().__init__(time, priority=1)
        self.vehicle = vehicle
        self.drop_off = drop_off

    def __repr__(self):
        return '{}_CompletesTrip@t{}'.format(self.vehicle, self.time)

    def trigger(self, end=False):
        if self.drop_off:
            Variables.EV_no -= 1  # Occupied vehicle becomes vacant
            new_ratio = self.vehicle.occupiedTime / (self.time - self.vehicle.assignedTime)
            self.vehicle.assignedTime = self.time
            self.vehicle.occupiedTime = 0

            # Update system EV utilisation with new vehicle (occupied : total) ratio
            Variables.utilisation = (Variables.total_trips * Variables.utilisation + new_ratio) / (Variables.total_trips + 1)
            Variables.total_trips += 1  # Passenger trip is marked as completed

            # Upon drop-off, HVs decide whether to leave the labour market
            # HV is forced to exit if drop-off occurs after the last passenger spawn (END of simulation)
            self.vehicle.decide_exit(self.time, end=end)

            # Record statistics
            Statistics.data_output['utilisation_data'].append([self.time,  self.vehicle.id,  new_ratio])
        else:  # Vehicle becomes available after recharging or repositioning
            EVs[self.vehicle.id] = self.vehicle
