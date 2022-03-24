from itertools import count
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from Configuration import configs
from Parser import depot_nodes, maximumWork
from Basics import Event, Location, random_loc, duration_between, Electricity
from Control import Parameters, Variables, Statistics


HVs = {}
activeAVs = {}
inactiveAVs = {}

# maximumWork = 24 * 3600  # Override maximum work hour limit
depot_dict = {Location(d): None for d in depot_nodes}


def load_vehicles(neoclassical=0.5):
    # Instantiate the total AV fleet as inactive at depots (chosen randomly)
    for d in np.random.choice(depot_nodes, configs['AV_fleet_size']):
        AV(0, Location(d))

    # Activate random AVs as the initial fleet
    ActivateAVs(0, configs['AV_initial_size'])

    # Load HVs (neoclassical or income-targeting) with preferred shift start time and duration
    total = configs['HV_fleet_size']
    morning = int(0.35 * total)
    afternoon = int(0.3 * total)
    evening = total - morning - afternoon

    # Preferred start times, note that they might be shifted by some demand pattern adjustments (-4)
    shift_start = []
    shift_start += [int(3600 * i) for i in truncnorm.rvs(-3, 3, 3, 1, morning)]
    shift_start += [int(3600 * i) for i in truncnorm.rvs(-2, 2, 9, 2, afternoon)]
    shift_start += [int(3600 * i) for i in truncnorm.rvs(-1, 1.5, 15, 2, evening)]
    # np.random.shuffle(shift_start)

    # Set HV supply prediction values
    bins = [i for i in range(0, Statistics.lastPassengerTime, configs['MPC_prediction_interval'])]
    Variables.histSupply, _ = np.histogram(shift_start, bins=bins)

    neoList = [k <= neoclassical for k in np.random.rand(total)]  # Proportion of neoclassical HVs

    hourlyCost = list(np.random.uniform(10, 40, total))
    # hourlyCost = list(truncnorm.rvs(a=-0.5, b=3, loc=18, scale=10, size=total))
    targetIncome = list(np.random.uniform(50, 300, total))

    for i in range(total):
        NewHV(shift_start[i], random_loc(), neoList[i], hourlyCost[i], targetIncome[i])

    # TODO: remove after supply test
    # pd.DataFrame({'start_times': shift_start, 'neoclassical': neoList}).to_csv('../Results/Simulation_Outputs/test_supply_data.csv')


def load_simple_vehicles():
    shift_start = [int(i) for i in np.linspace(0, 3600, configs['HV_fleet_size'])]
    initialSoC = np.random.uniform(0.9, 1.0, configs['HV_fleet_size'])
    bins = [i for i in range(0, Statistics.lastPassengerTime, configs['MPC_prediction_interval'])]
    Variables.histSupply = pd.Series(shift_start).groupby(pd.cut(pd.Series(shift_start), bins)).count()

    # Activate all HV fleet as
    for i in range(configs['HV_fleet_size']):
        # NewHV(shift_start[i], random_loc(), False, 0, 10000)
        NewEV(shift_start[i], random_loc(), False, 0, 10000, initialSoC[i])


class Vehicle:
    _ids = count(0)

    def __init__(self, time, loc):
        self.id = next(self._ids)
        self.time = time
        self.loc = loc
        self.is_HV = None
        self.income = 0

        self.entranceTime = time
        self.assignedTime = 0
        self.occupiedTime = 0

        self.nextTrip = None  # TripCompletion object, checked at planned destination
        self.destination = None  # Planned cruise destination, updated if intercepted by trip assignment


class HV(Vehicle):
    def __init__(self, time, loc, neo, hourlyCost, targetIncome):
        super().__init__(time, loc)
        self.is_HV = True
        HVs[self.id] = self  # Instantiated as vacant HV

        # Driver behavioural attributes
        self.neoclassical = neo
        self.hourlyCost = hourlyCost
        self.targetIncome = targetIncome

        # Record statistics
        Statistics.vehicle_data = Statistics.vehicle_data.append({'v_id': self.id,
                                                                  'is_HV': True,
                                                                  'neoclassical': self.neoclassical,
                                                                  'income': self.income,
                                                                  'time': self.time,
                                                                  'activation': True}, ignore_index=True)

        # Update market count statistics
        Variables.HV_total += 1

    def __repr__(self):
        return 'HV{}'.format(self.id)

    def decide_exit(self, exitTime, end=False):
        if exitTime - self.entranceTime >= maximumWork or end:
            # Update market count statistics
            Variables.HV_total -= 1

            # Force exit labour market and record statistics
            Statistics.vehicle_data = Statistics.vehicle_data.append({'v_id': self.id,
                                                                      'is_HV': True,
                                                                      'neoclassical': self.neoclassical,
                                                                      'income': self.income,
                                                                      'time': exitTime,
                                                                      'activation': False}, ignore_index=True)
        else:
            # if self.neoclassical and (Variables.HV_wage * Parameters.HV_utilisation >= self.hourlyCost):
            # if self.neoclassical and (Variables.HV_wage * Parameters.HV_occupancy >= self.hourlyCost):
            if self.neoclassical and ((self.hourlyCost - Variables.HV_wage * Parameters.HV_occupancy) / self.hourlyCost < np.random.rand() - 0.2):
                Variables.exitDecisions += 1  # Update number of exit decision-making

                # Neoclassical drivers have a chance to continue working based on the expected wage
                HVs[self.id] = self
            elif ~self.neoclassical and (self.income < self.targetIncome):
                # Income-targeting drivers continue to work if accumulated income < target income
                HVs[self.id] = self
            else:
                # Update market count statistics
                Variables.HV_total -= 1

                # Exit labour market and record statistics
                Statistics.vehicle_data = Statistics.vehicle_data.append({'v_id': self.id,
                                                                          'is_HV': True,
                                                                          'neoclassical': self.neoclassical,
                                                                          'income': self.income,
                                                                          'time': exitTime,
                                                                          'activation': False}, ignore_index=True)


class AV(Vehicle):
    def __init__(self, time, loc):
        super().__init__(time, loc)
        self.is_HV = False
        inactiveAVs[self.id] = self  # AVs are loaded as inactive

    def __repr__(self):
        return 'AV{}'.format(self.id)

    def activate(self):
        assert self.id in inactiveAVs, 'Cannot activate an already active AV_{}'.format(self.id)

        activeAVs[self.id] = inactiveAVs.pop(self.id)
        Variables.AV_total += 1

        # Record statistics
        Statistics.vehicle_data = Statistics.vehicle_data.append({'v_id': self.id,
                                                                  'is_HV': False,
                                                                  'neoclassical': None,
                                                                  'income': self.income,
                                                                  'time': self.time,
                                                                  'activation': True}, ignore_index=True)

    # A vacant AV can be deactivated by going to the nearest depot for charging and parking
    def deactivate(self):
        assert self.id in activeAVs, 'Cannot deactivate non-vacant AV_{}'.format(self.id)

        # Dictionary of travel time to each depot location
        for d in depot_dict.keys():
            depot_dict[d] = duration_between(self.loc, d)

        depot = min(depot_dict, key=depot_dict.get)
        tt = depot_dict[depot]

        # AV travels to the nearest depot
        self.time += tt
        self.loc = depot

        inactiveAVs[self.id] = activeAVs.pop(self.id)
        Variables.AV_total -= 1

        # Record statistics
        Statistics.vehicle_data = Statistics.vehicle_data.append({'v_id': self.id,
                                                                  'is_HV': False,
                                                                  'neoclassical': None,
                                                                  'income': self.income,
                                                                  'time': self.time,
                                                                  'activation': False}, ignore_index=True)


class TripCompletion(Event):
    def __init__(self, time, vehicle, drop_off=False):
        super().__init__(time, priority=1)
        self.vehicle = vehicle
        self.drop_off = drop_off

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return '{}_CompletesTrip@t{}'.format(self.vehicle, self.time)

    def trigger(self, end=False):
        if self.drop_off:
            newRatio = self.vehicle.occupiedTime / (self.time - self.vehicle.assignedTime)
            self.vehicle.assignedTime = self.time
            self.vehicle.occupiedTime = 0

            if self.vehicle.is_HV:
                # Update system HV utilisation with new vehicle (occupied : total) ratio
                Parameters.HV_utilisation = (Variables.HV_trips * Parameters.HV_utilisation + newRatio) / (Variables.HV_trips + 1)
                Variables.HV_trips += 1

                # Upon drop-off, HVs decide whether to leave the labour market
                # HV is forced to exit if drop-off occurs after the last passenger spawn (END of simulation)
                self.vehicle.decide_exit(self.time, end=end)
            else:
                # Update system AV utilisation
                Parameters.AV_utilisation = (Variables.AV_trips * Parameters.AV_utilisation + newRatio) / (Variables.AV_trips + 1)
                Variables.AV_trips += 1

                activeAVs[self.vehicle.id] = self.vehicle

            # Record statistics
            Statistics.utilisation_data = Statistics.utilisation_data.append({'time': self.time,
                                                                              'v_id': self.vehicle.id,
                                                                              'trip_utilisation': newRatio}, ignore_index=True)


class ActivateAVs(Event):
    def __init__(self, time, size):
        super().__init__(time, priority=0)
        # NOTE: activation cannot exceed the total fleet size
        self.size = min(size, len(inactiveAVs))

    def __repr__(self):
        return 'Activate_{}AVs@t{}'.format(self.size, self.time)

    def trigger(self):
        for v in np.random.choice(list(inactiveAVs.values()), self.size, False):
            v.time = self.time  # Update vehicle time
            v.activate()


class DeactivateAVs(Event):
    def __init__(self, time, size):
        super().__init__(time, priority=0)
        self.size = size

    def __repr__(self):
        return 'Deactivate_{}AVs@t{}'.format(self.size, self.time)

    def trigger(self):
        if (self.size > len(activeAVs)) or (len(activeAVs) == 0):
            for v in activeAVs.copy().values():  # Deactivate all active AVs
                # v.update_loc(self.time)
                v.time = self.time
                v.deactivate()

            if self.time < Statistics.lastPassengerTime:
                DeactivateAVs(self.time+1, self.size-len(activeAVs))  # Try to deactivate the remaining AVs at next second
        else:
            for v in np.random.choice(list(activeAVs.values()), self.size, False):
                # v.update_loc(self.time)
                v.time = self.time
                v.deactivate()


class EV(HV):
    def __init__(self, time, loc, neo, hourlyCost, targetIncome, initialSoC):
        super().__init__(time, loc, neo, hourlyCost, targetIncome)
        self.SoC = initialSoC * Electricity.max_SoC

    def __repr__(self):
        return 'EV{}'.format(self.id)


class NewHV(Event):

    def __init__(self, time, loc, neo, hourlyCost, targetIncome):
        super().__init__(time, priority=0)
        self.loc = loc
        self.neo = neo
        self.hourlyCost = hourlyCost
        self.targetIncome = targetIncome

    def __repr__(self):
        return 'NewHV@t{}'.format(self.time)

    def trigger(self):
        # Income-targeting drivers always start work to make an income
        # Neoclassical drivers start work if expected revenue >= hourly cost at start times
        # expectedWage = Variables.HV_wage * Parameters.HV_utilisation
        expectedWage = Variables.HV_wage * Parameters.HV_occupancy

        if ~self.neo or (expectedWage >= self.hourlyCost):
            HV(self.time, self.loc, self.neo, self.hourlyCost, self.targetIncome)
        elif self.neo and (self.time + 600 < Statistics.lastPassengerTime) and \
                ((self.hourlyCost - expectedWage) / self.hourlyCost < np.random.rand() - 0.2):
            # Neoclassical drivers may try to join the market again in 5 minutes (before last passenger)
            NewHV(self.time + 300, self.loc, self.neo, self.hourlyCost, self.targetIncome)
            Variables.histSupply[int((self.time + 300) / 10)] += 1


class NewEV(Event):

    def __init__(self, time, loc, neo, hourlyCost, targetIncome, initialSoC):
        super().__init__(time, priority=0)
        self.loc = loc
        self.neo = neo
        self.hourlyCost = hourlyCost
        self.targetIncome = targetIncome
        self.initialSoC = initialSoC

    def __repr__(self):
        return 'NewEV@t{}'.format(self.time)

    def trigger(self):
        expectedWage = Variables.HV_wage * Parameters.HV_occupancy

        if ~self.neo or (expectedWage >= self.hourlyCost):
            EV(self.time, self.loc, self.neo, self.hourlyCost, self.targetIncome, self.initialSoC)
        elif self.neo and (self.time + 600 < Statistics.lastPassengerTime) and \
                ((self.hourlyCost - expectedWage) / self.hourlyCost < np.random.rand() - 0.2):
            # Neoclassical drivers may try to join the market again in 5 minutes (before last passenger)
            NewHV(self.time + 300, self.loc, self.neo, self.hourlyCost, self.targetIncome)
            Variables.histSupply[int((self.time + 300) / 10)] += 1
