from itertools import count
from numpy import random
import bisect

from Configuration import configs
from Parser import depot_nodes
from Map import G
from Basics import Event, Location, random_loc, path_between, duration_between
from Control import vehicle_data


HVs = {}
activeAVs = {}
inactiveAVs = {}
depot_dict = {Location(d): None for d in depot_nodes}


def load_vehicles():
    # Instantiate the total AV fleet as inactive at depots (chosen randomly)
    for d in random.choice(depot_nodes, configs['AV_fleet_size']):
        AV(0, Location(d))

    # Activate initial AV fleet
    ActivateAVs(0, configs['AV_initial_size'])

    # Instantiate HV fleet at random locations at random times within a span (e.g. the first 10 min)
    for t in random.randint(0, configs['load_vehicle_span'], configs['HV_fleet_size']):
        NewHV(t)


class Vehicle:
    _ids = count(0)

    def __init__(self, time, loc):
        self.id = next(self._ids)
        self.time = time
        self.loc = loc
        self.is_HV = None
        self.nextCruise = None
        self.destination = None
        self.pathNodes = None  # Path from Basics.path_between(), including the upstream intersection as the first node
        self.pathTimes = None  # Timestamps to reach path nodes, the first timestamp is the current time

    def cruise(self, plan='random_destination'):
        # Cruise to a random intersection
        if plan == 'random_destination':
            self.destination = Location(random.choice(G.nodes()))
            pathNodes = [self.loc.source] + path_between(self.loc, self.destination)

            timestamp = self.time + self.loc.timeFromTarget
            pathTimes = [self.time, timestamp]
            for i in range(1, len(pathNodes) - 1):
                timestamp += G.edges[pathNodes[i], pathNodes[i + 1]]['travel_time']
                pathTimes.append(timestamp)

            if self.loc.type == 'Intersection':
                pathNodes.pop()
                pathTimes.pop()

            self.pathNodes = pathNodes
            self.pathTimes = pathTimes
            return pathTimes[-1]
        else:
            raise ValueError('Cannot recognise cruising plan.')

    def update_loc(self, t):
        if t == self.pathTimes[0]:  # Vehicle is at the beginning
            pass
        elif t in self.pathTimes:  # Vehicle is at an intersection
            self.time = t
            self.loc = Location(self.pathNodes[self.pathTimes.index(t)])
        else:  # Vehicle is on a road, between intersections
            idx = bisect.bisect_left(self.pathTimes, t)
            if idx == 1:  # current time is before reaching the first intersection
                n0 = self.pathNodes[0]
                n1 = self.pathNodes[1]

                deltaT = t - self.time
                self.loc.timeFromTarget -= deltaT
                self.loc.timeFromSource += deltaT

                deltaD = deltaT * G.edges[n0, n1]['length'] / G.edges[n0, n1]['travel_time']
                self.loc.locFromTarget -= deltaD
                self.loc.locFromSource += deltaD
            elif idx < len(self.pathTimes):
                originalNodes = self.pathNodes
                originalTimes = self.pathTimes
                self.pathNodes = self.pathNodes[idx-1::]
                self.pathTimes = self.pathTimes[idx-1::]
                if (len(self.pathNodes) <= 1) or (len(self.pathTimes) <= 1):  # TODO: debug
                    print(originalTimes)
                    print(self.pathTimes)
                    print(originalNodes)
                    print(self.pathNodes)
                    raise ValueError('t{} inserted at idx {} exceeds the planned cruise path times'.format(t, idx))

                n0 = self.pathNodes[0]
                n1 = self.pathNodes[1]
                pos = (t - self.pathTimes[0]) * G.edges[n0, n1]['length'] / G.edges[n0, n1]['travel_time']
                self.loc = Location(n0, n1, pos)

            self.time = t  # Update time


class HV(Vehicle):
    def __init__(self, time, loc):
        super().__init__(time, loc)
        self.is_HV = True
        HVs[self.id] = self
        self.nextCruise = CruiseTrip(self.cruise(), self)

    def __repr__(self):
        return 'HV{}'.format(self.id)

    def exit(self, exitTime):
        # TODO: Market exit (priority 0)
        # del self.nextCruise
        del HVs[self.id]

        # Record data ['v_id', 'is_HV', 'time', 'activation']
        vehicle_data.append([self.id, self.is_HV, exitTime, False])


class AV(Vehicle):
    def __init__(self, time, loc):
        super().__init__(time, loc)
        self.is_HV = False
        inactiveAVs[self.id] = self

    def __repr__(self):
        return 'AV{}'.format(self.id)

    def activate(self):
        if self.id in inactiveAVs:
            self.nextCruise = CruiseTrip(self.cruise(), self)
            activeAVs[self.id] = inactiveAVs.pop(self.id)
        else:
            raise KeyError('Cannot activate an already active AV_{}'.format(self.id))

    # A vacant AV can be deactivated by going to the nearest depot for charging and parking
    def deactivate(self):
        if self.id in activeAVs:  # Vacant AVs
            # Dictionary of travel time to each depot location
            for d in depot_dict.keys():
                depot_dict[d] = duration_between(self.loc, d)

            depot = min(depot_dict, key=depot_dict.get)
            tt = depot_dict[depot]

            # AV travels to the nearest depot
            self.time += tt
            self.loc = depot

            inactiveAVs[self.id] = activeAVs.pop(self.id)
        else:
            raise KeyError('Cannot deactivate non-vacant AV_{}'.format(self.id))


class CruiseTrip(Event):
    def __init__(self, time, vehicle, drop_off=False):
        super().__init__(time, priority=1)
        self.vehicle = vehicle
        self.drop_off = drop_off
        self.endTime = 10800  # TODO: manually setting simulation end time

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return '{}_Cruise@t{}'.format(self.vehicle, self.time)

    def trigger(self):
        # Add vehicle back into vacancy dictionary upon drop-off
        if self.drop_off:
            if self.vehicle.is_HV:
                HVs[self.vehicle.id] = self.vehicle
            else:
                activeAVs[self.vehicle.id] = self.vehicle

        if self.vehicle.nextCruise is self:
            # Reaching the planned cruising destination without assignment
            self.vehicle.time = self.time
            self.vehicle.loc = self.vehicle.destination

            # Cruise to the next destination if simulation has not ended
            if self.time < self.endTime:
                nextTime = self.vehicle.cruise()
                self.vehicle.nextCruise = CruiseTrip(nextTime, self.vehicle)


class ActivateAVs(Event):
    # TODO: catch edge case when activate number exceeds the total available fleet size
    def __init__(self, time, size):
        super().__init__(time, priority=0)
        self.size = size

    def __repr__(self):
        return 'Activate_{}AVs@t{}'.format(self.size, self.time)

    def trigger(self):
        print('Activate {} AVs at time {}'.format(self.size, self.time))
        for v in random.choice(list(inactiveAVs.values()), self.size, False):
            v.time = self.time  # Update vehicle time
            v.activate()

            # Record data ['v_id', 'is_HV', 'time', 'activation']
            vehicle_data.append([v.id, v.is_HV, self.time, True])


class DeactivateAVs(Event):
    # TODO: behaviour when instantaneous vacant AVs are fewer than optimal deactivation size
    def __init__(self, time, size):
        super().__init__(time, priority=0)
        self.size = size

    def __repr__(self):
        return 'Deactivate_{}AVs@t{}'.format(self.size, self.time)

    def trigger(self):
        if (self.size > len(activeAVs)) or (len(activeAVs) == 0):
            # Cannot deactivate more than current vacant vehicles, or if there are no vacant vehicles
            DeactivateAVs(self.time+1, self.size)  # Delay deactivation by 1 sec
        else:
            print('Deactivate {} AVs at time {}'.format(self.size, self.time))
            for v in random.choice(list(activeAVs.values()), self.size, False):
                v.update_loc(self.time)
                v.deactivate()

                # Record data ['v_id', 'is_HV', 'time', 'activation']
                vehicle_data.append([v.id, v.is_HV, self.time, False])


class NewHV(Event):
    def __init__(self, time):
        super().__init__(time, priority=0)

    def __repr__(self):
        return 'NewHV@t{}'.format(self.time)

    def trigger(self):
        v = HV(self.time, random_loc())

        # Record data ['v_id', 'is_HV', 'time', 'activation']
        vehicle_data.append([v.id, v.is_HV, self.time, True])
