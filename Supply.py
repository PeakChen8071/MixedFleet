from itertools import count
import numpy as np
from scipy.stats import truncnorm

from Configuration import configs
from Parser import depot_nodes
from Basics import Event, Location, random_loc, duration_between
from Control import Variables, vehicle_data, utilisation_data


HVs = {}
maximumWork = configs['maximum_work_duration']

activeAVs = {}
inactiveAVs = {}
cruiseAV = configs['AV_cruise_mode']
depot_dict = {Location(d): None for d in depot_nodes}


def load_vehicles():
    # Instantiate the total AV fleet as inactive at depots (chosen randomly)
    for d in np.random.choice(depot_nodes, configs['AV_fleet_size']):
        AV(0, Location(d))

    # Activate random AVs as the initial fleet
    ActivateAVs(0, configs['AV_initial_size'])

    # Load HVs (neoclassical or income-targeting) with preferred shift start time and duration
    total = configs['HV_fleet_size']
    morning = int(0.55 * total)
    afternoon = int(0.15 * total)
    evening = total - morning - afternoon
    
    # Preferred start times, note that they might be shifted by some demand pattern adjustments (-4)
    shift_start = []
    shift_start += [int(3600 * i) for i in truncnorm.rvs(-2, 2, 3, 1, morning)]
    shift_start += [int(3600 * i) for i in truncnorm.rvs(-2, 2, 9, 1, afternoon)]
    shift_start += [int(3600 * i) for i in truncnorm.rvs(-2, 2, 15, 1, evening)]
    # np.random.shuffle(shift_start)

    neoclassical = [k <= 0.5 for k in np.random.rand(total)]
    # neoclassical = [False] * total
    hourlyCost = list(np.random.uniform(10, 25, total))
    targetIncome = list(np.random.uniform(50, 200, total))

    for i in range(total):
        NewHV(shift_start[i], random_loc(), neoclassical[i], hourlyCost[i], targetIncome[i])


class Vehicle:
    _ids = count(0)

    def __init__(self, time, loc):
        self.id = next(self._ids)
        self.time = time
        self.loc = loc
        self.is_HV = None
        self.shiftStartTime = time
        self.tripStartTime = time
        self.occupiedTime = 0

        # Cruise-related attributes
        self.nextTrip = None  # TripCompletion object, checked at planned destination
        self.destination = None  # Planned cruise destination, updated if intercepted by trip assignment
        self.pathNodes = None  # Path from Basics.path_between(), including the upstream intersection as the first node
        self.pathTimes = None  # Timestamps to reach path nodes, the first timestamp is the current time

    # NOTE: move cruise() method to HV/AV subclass methods if their behaviours are significantly different
    # NOTE: End cruise at simulation end time to prevent infinite cruising
    # def cruise(self, random_destination=True):
    #     assert random_destination, 'Cannot recognise cruising plan.'

    #     self.destination = Location(np.random.choice(G.nodes()))
    #     pathNodes = [self.loc.source] + path_between(self.loc, self.destination)
    #
    #     timestamp = self.time + self.loc.timeFromTarget
    #     pathTimes = [self.time, timestamp]
    #     for i in range(1, len(pathNodes) - 1):
    #         timestamp += G.edges[pathNodes[i], pathNodes[i + 1]]['duration']
    #         pathTimes.append(timestamp)
    #
    #     if self.loc.type == 'Intersection':
    #         pathNodes.pop()
    #         pathTimes.pop()
    #
    #     self.pathNodes = pathNodes
    #     self.pathTimes = pathTimes
    #     return pathTimes[-1]

    # update_loc() calculates the location of vehicle at time t, along its current cruising path.
    # def update_loc(self, t):
    #     if not self.pathTimes or t == self.pathTimes[0]:  # Vehicle is at the beginning
    #         pass
    #     elif t in self.pathTimes:  # Vehicle is at an intersection
    #         self.time = t
    #         self.loc = Location(self.pathNodes[self.pathTimes.index(t)])
    #     else:  # Vehicle is on a road, between intersections
    #         idx = bisect.bisect_left(self.pathTimes, t)
    #         if idx == 1:  # current time is before reaching the first intersection
    #             n0 = self.pathNodes[0]
    #             n1 = self.pathNodes[1]
    #
    #             deltaT = t - self.time
    #             self.loc.timeFromTarget -= deltaT
    #             self.loc.timeFromSource += deltaT
    #
    #             deltaD = deltaT * G.edges[n0, n1]['distance'] / G.edges[n0, n1]['duration']
    #             self.loc.locFromTarget -= deltaD
    #             self.loc.locFromSource += deltaD
    #         elif idx < len(self.pathTimes):
    #             self.pathNodes = self.pathNodes[idx-1::]
    #             self.pathTimes = self.pathTimes[idx-1::]
    #
    #             n0 = self.pathNodes[0]
    #             n1 = self.pathNodes[1]
    #             pos = (t - self.pathTimes[0]) * G.edges[n0, n1]['distance'] / G.edges[n0, n1]['duration']
    #             self.loc = Location(n0, n1, pos)


class HV(Vehicle):
    def __init__(self, time, loc, neoclassical, hourlyCost, targetIncome):
        super().__init__(time, loc)
        self.is_HV = True
        self.income = 0
        HVs[self.id] = self  # Instantiated as vacant HV

        # Driver behavioural attributes
        self.neoclassical = neoclassical
        self.hourlyCost = hourlyCost
        self.targetIncome = targetIncome

        # Record data ['v_id', 'is_HV', 'income', 'time', 'activation']
        vehicle_data.append([self.id, True, self.income, self.time, True])

    def __repr__(self):
        return 'HV{}'.format(self.id)

    def decide_exit(self, exitTime, force=False):
        if exitTime - self.shiftStartTime >= maximumWork or force:
            # Force exit labour market and record data ['v_id', 'is_HV', 'income', 'time', 'activation']
            vehicle_data.append([self.id, True, self.income, exitTime, False])
        else:
            if self.neoclassical and (Variables.unitWage * Variables.HV_utilisation >= self.hourlyCost / 3600):
                # Neoclassical drivers continue to work if current unit wage >= unit cost
                HVs[self.id] = self
            elif ~self.neoclassical and (self.income < self.targetIncome):
                # Income-targeting drivers continue to work if accumulated income < target income
                HVs[self.id] = self
            else:
                # Exit labour market and record data ['v_id', 'is_HV', 'income', 'time', 'activation']
                vehicle_data.append([self.id, True, self.income, exitTime, False])


class AV(Vehicle):
    def __init__(self, time, loc):
        super().__init__(time, loc)
        self.is_HV = False
        inactiveAVs[self.id] = self  # AVs are loaded as inactive

    def __repr__(self):
        return 'AV{}'.format(self.id)

    def activate(self):
        assert self.id in inactiveAVs, 'Cannot activate an already active AV_{}'.format(self.id)

        # if cruiseAV:
        #     self.nextTrip = TripCompletion(self.time, self)
        activeAVs[self.id] = inactiveAVs.pop(self.id)

        # Record data ['v_id', 'is_HV', 'income', 'time', 'activation']
        vehicle_data.append([self.id, False, 0, self.time, True])

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
            newRatio = self.vehicle.occupiedTime / (self.time - self.vehicle.tripStartTime)
            self.vehicle.tripStartTime = self.time
            self.vehicle.occupiedTime = 0

            if self.vehicle.is_HV:
                # Update system HV utilisation with new vehicle (occupied : total) ratio
                Variables.HV_utilisation = (Variables.HV_trips * Variables.HV_utilisation + newRatio) / (Variables.HV_trips + 1)
                Variables.HV_trips += 1

                # Upon drop-off, HVs decide whether to leave the labour market
                # HV is forced to exit if drop-off occurs after the last passenger spawn (END of simulation)
                self.vehicle.decide_exit(self.time, force=end)
            else:
                # Update system AV utilisation
                Variables.AV_utilisation = (Variables.AV_trips * Variables.AV_utilisation + newRatio) / (Variables.AV_trips + 1)
                Variables.AV_trips += 1

                if not end:  # Back to vacant state
                    activeAVs[self.vehicle.id] = self.vehicle

            # Record utilisation data ['delivery_t', 'v_id', 'trip_utilisation']
            utilisation_data.append([self.time, self.vehicle.id, newRatio])

        # if self.vehicle.nextTrip is self:
        #     # Reaching the planned cruising destination without assignment
        #     self.vehicle.time = self.time
        #     self.vehicle.loc = self.vehicle.destination
        #
        #     # Cruise to the next destination if simulation has not ended
        #     if self.time < self.endTime:
        #         nextTime = self.vehicle.cruise()
        #         self.vehicle.nextTrip = TripCompletion(nextTime, self.vehicle)


class ActivateAVs(Event):
    def __init__(self, time, size):
        super().__init__(time, priority=0)
        # ASSUMPTION: activation cannot exceed the total fleet size
        self.size = min(size, len(inactiveAVs))

    def __repr__(self):
        return 'Activate_{}AVs@t{}'.format(self.size, self.time)

    def trigger(self):
        print('Activate {} AVs at time {}'.format(self.size, self.time))
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
        # TODO: behaviour when instantaneous vacant AVs are fewer than optimal deactivation size
        if (self.size > len(activeAVs)) or (len(activeAVs) == 0):
            # Cannot deactivate more than current vacant vehicles, or if there are no vacant vehicles
            DeactivateAVs(self.time+1, self.size)  # Delay deactivation by 1 sec
        else:
            print('Deactivate {} AVs at time {}'.format(self.size, self.time))
            for v in np.random.choice(list(activeAVs.values()), self.size, False):
                # v.update_loc(self.time)
                v.time = self.time
                v.deactivate()

                # Record data ['v_id', 'is_HV', 'income', 'time', 'activation']
                vehicle_data.append([v.id, False, 0, self.time, False])


class NewHV(Event):
    def __init__(self, time, loc, neoclassical, hourlyCost, targetIncome):
        super().__init__(time, priority=0)
        self.loc = loc
        self.neoclassical = neoclassical
        self.hourlyCost = hourlyCost
        self.targetIncome = targetIncome

    def __repr__(self):
        return 'NewHV@t{}'.format(self.time)

    def trigger(self):
        # Income-targeting drivers always start work to make an income
        # Neoclassical drivers start work if expected revenue >= hourly cost at start times
        if ~self.neoclassical or (Variables.unitWage * Variables.HV_utilisation >= self.hourlyCost / 3600):
            HV(self.time, self.loc, self.neoclassical, self.hourlyCost, self.targetIncome)
        elif self.neoclassical and (1 / (1 + np.exp(0.2 * (Variables.unitWage * 3600 * Variables.HV_utilisation - self.hourlyCost))) < 0.5):
            # Neoclassical drivers may try to join the market again in 5 minutes with binary Logit choice
            NewHV(self.time + 300, self.loc, self.neoclassical, self.hourlyCost, self.targetIncome)
