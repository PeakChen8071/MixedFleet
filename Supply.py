from itertools import count
import numpy as np

from Configuration import configs
from Parser import depot_nodes, maximumWork
from Basics import Event, Location, random_loc, duration_between, population_start_time
from Control import Parameters, Variables, Statistics


HVs = {}
activeAVs = {}
inactiveAVs = {}

depot_dict = {Location(d): None for d in depot_nodes}


def load_vehicles(neoclassical=0.5):
    # Instantiate the total AV fleet as inactive at depots (chosen randomly)
    for d in np.random.choice(depot_nodes, configs['AV_fleet_size']):
        AV(0, Location(d))

    # Activate random AVs as the initial fleet
    ManageAVs(0, configs['AV_initial_size'])

    neoList = [k <= neoclassical for k in np.random.rand(configs['HV_fleet_size'])]  # Proportion of neoclassical HVs

    hourlyCost = list(np.random.uniform(10, 24, len(population_start_time)))
    # hourlyCost = list(truncnorm.rvs(a=-0.5, b=3, loc=18, scale=10, size=total))
    targetIncome = list(np.random.uniform(100, 200, len(population_start_time)))

    for i in range(len(population_start_time)):
        # NewHV(shift_start[i], random_loc(), neoList[i], hourlyCost[i], targetIncome[i])
        NewHV(population_start_time[i], random_loc(), neoList[i], hourlyCost[i], targetIncome[i])


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
        Statistics.vehicle_data.append([self.id, True, self.neoclassical, self.hourlyCost, self.targetIncome,
                                        self.income, self.time, True])

        # Update market count statistics
        Variables.HV_total += 1

    def __repr__(self):
        return 'HV{}'.format(self.id)

    def decide_exit(self, exitTime, end=False):
        Variables.exitDecisions += 1  # Update number of exit decision-making

        if (exitTime - self.entranceTime >= maximumWork) or end:
            # Update market count statistics
            Variables.HV_total -= 1
            Variables.totalWage -= self.income

            # Force exit labour market and record statistics
            Statistics.vehicle_data.append([self.id, True, self.neoclassical, self.hourlyCost, self.targetIncome,
                                            self.income, exitTime, False])
        else:
            # if self.neoclassical and (Variables.HV_wage * Parameters.HV_utilisation >= self.hourlyCost):
            # if self.neoclassical and ((self.hourlyCost - Variables.HV_wage * Parameters.HV_occupancy) / self.hourlyCost < np.random.rand()):
            if self.neoclassical and (0.5 - (Variables.HV_wage * Parameters.HV_occupancy - self.hourlyCost) / (2 * np.sqrt(1 + (Variables.HV_wage * Parameters.HV_occupancy - self.hourlyCost) ** 2)) < np.random.rand()):
                # Neoclassical drivers have a chance to continue working based on the expected gain
                HVs[self.id] = self
            elif ~self.neoclassical and (self.income < self.targetIncome):
                # Income-targeting drivers continue to work if accumulated income has not reached the target
                HVs[self.id] = self
            else:
                # Update market count statistics
                Variables.HV_total -= 1
                Variables.totalWage -= self.income

                # Exit labour market and record statistics
                Statistics.vehicle_data.append([self.id, True, self.neoclassical, self.hourlyCost, self.targetIncome,
                                                self.income, exitTime,  False])


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
        Statistics.vehicle_data.append([self.id, False, None, None, None, self.income, self.time, True])

    # A vacant AV can be deactivated. It remains stationary at its last parked location
    def deactivate(self):
        assert self.id in activeAVs, 'Cannot deactivate non-vacant AV_{}'.format(self.id)

        inactiveAVs[self.id] = activeAVs.pop(self.id)
        Variables.AV_total -= 1

        # Record statistics
        Statistics.vehicle_data.append([self.id, False, None, None, None, self.income, self.time, False])


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
            Statistics.utilisation_data.append([self.time, self.vehicle.id, newRatio])


# Dynamic AV fleet management: activation/deactivation
class ManageAVs(Event):
    def __init__(self, time, size):
        super().__init__(time, priority=0)
        if size >= 0:  # Activation
            self.size = size
            self.activation = True
        elif size < 0:  # Deactivation
            self.size = -size
            self.activation = False

    def __repr__(self):
        if self.activation:
            return 'Activate_{}AVs@t{}'.format(self.size, self.time)
        else:
            return 'Deactivate_{}AVs@t{}'.format(self.size, self.time)

    def trigger(self):
        if self.activation:  # Activation
            if self.size > len(inactiveAVs):  # Activation cannot exceed the reserve fleet size
                print('WARNING! Number of activation exceeds total inactive AVs')

            for v in np.random.choice(list(inactiveAVs.values()), min(self.size, len(inactiveAVs)), replace=False):
                v.time = self.time  # Update vehicle time
                v.activate()
        else:  # Deactivation
            if self.size > len(activeAVs):  # If the deactivation exceeds number of current vacant AVs
                for v in activeAVs.copy().values():  # Deactivate all active AVs
                    v.time = self.time
                    v.deactivate()

                if self.time < Statistics.lastPassengerTime:
                    # Try to deactivate the remaining AVs at next second
                    ManageAVs(self.time + 1, len(activeAVs) - self.size)
            else:
                for v in np.random.choice(list(activeAVs.values()), self.size, replace=False):
                    v.time = self.time
                    v.deactivate()


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
        HV(self.time, self.loc, self.neo, self.hourlyCost, self.targetIncome)
