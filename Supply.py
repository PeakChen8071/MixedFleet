from itertools import count
from numpy.random import randint
import bisect

from Configuration import configs
from Map import G
from Basics import Event, Location, random_loc, path_between
from Control import vehicle_data


HVs = {}
AVs = {}
parkedAVs = {}


def load_vehicles():
    for t1 in randint(0, configs['load_vehicle_span'], configs['HV_fleet_size']):
        NewVehicle(t1, True)
    for t2 in randint(0, configs['load_vehicle_span'], configs['AV_fleet_size']):
        NewVehicle(t2, False)


class Vehicle:
    _ids = count(0)

    def __init__(self, time, loc):
        self.id = next(self._ids)
        self.time = time
        self.loc = loc
        self.is_HV = None
        self.nextCruise = None
        self.pathNodes = None
        self.pathTimes = None
        self.destination = None

    def update_loc(self, t):
        idx = bisect.bisect_left(self.pathTimes, t)
        if idx == 0:  # Before the first intersection
            deltaT = t - self.time
            deltaD = deltaT * self.loc.length / self.loc.travelTime
            self.loc.timeFromTarget -= deltaT
            self.loc.timeFromSource += deltaT
            self.loc.locFromTarget -= deltaD
            self.loc.locFromSource += deltaD
        else:
            if idx >= len(self.pathTimes):
                print(idx)
                print(t)
                print(self.pathTimes)
                print(self.pathNodes)

            self.pathNodes = self.pathNodes[idx-1::]
            self.pathTimes = self.pathTimes[idx-1::]
            source = self.pathNodes[0]
            target = self.pathNodes[1]
            pos = (t - self.pathTimes[0]) * G.edges[source, target]['length'] / G.edges[source, target]['travel_time']
            self.loc = Location(source, target, pos)

        self.time = t

    def cruise(self, plan='random_destination'):
        if plan == 'random_destination':
            self.destination = random_loc()
            pathNodes = path_between(self.loc, self.destination)
            timestamp = self.time
            pathTimes = [timestamp]

            for i in range(len(pathNodes) - 1):
                timestamp += G.edges[pathNodes[i], pathNodes[i + 1]]['travel_time']
                pathTimes.append(timestamp)

            if self.destination != 'Intersection':
                timestamp += self.destination.timeFromSource
                pathTimes.append(timestamp)
                pathNodes.append(self.destination.target)

            self.pathNodes = pathNodes
            self.pathTimes = pathTimes
            return timestamp
        else:
            raise ValueError('Cannot recognise cruising plan.')


class HV(Vehicle):
    def __init__(self, time, loc):
        super().__init__(time, loc)
        self.is_HV = True
        HVs[self.id] = self
        self.nextCruise = CruiseTrip(self.cruise(), self)

    def __repr__(self):
        return 'HV{}'.format(self.id)

    def exit(self):
        pass


class AV(Vehicle):
    def __init__(self, time, loc):
        super().__init__(time, loc)
        self.is_HV = False
        AVs[self.id] = self
        self.nextCruise = CruiseTrip(self.cruise(), self)

    def __repr__(self):
        return 'AV{}'.format(self.id)

    def activate(self):
        if self.id in AVs:
            raise KeyError('AV is already active.')
        else:
            AVs[self.id] = self

    def deactivate(self):
        # TODO: Go to nearest parking facility for charging and maintenance
        if self.id in AVs:
            parkedAVs[self.id] = AVs.pop(self.id)
        else:
            raise KeyError('Cannot deactivate non-vacant AVs.')


class NewVehicle(Event):
    def __init__(self, time, new_HV):
        super().__init__(time, priority=0)
        self.new_HV = new_HV

    def __repr__(self):
        if self.new_HV:
            return 'NewHV_@t{}'.format(self.time)
        else:
            return 'NewAV_@t{}'.format(self.time)

    def trigger(self):
        if self.new_HV:
            v = HV(self.time, random_loc())
        else:
            v = AV(self.time, random_loc())

        vehicle_data['is_HV'].update({v.id: self.new_HV})
        vehicle_data['deploy_t'].update({v.id: self.time})


class CruiseTrip(Event):
    def __init__(self, time, vehicle, drop_off=False):
        super().__init__(time, priority=0)
        self.vehicle = vehicle
        self.drop_off = drop_off

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return '{}_Cruise@t{}'.format(self.vehicle, self.time)

    def trigger(self, end=False):
        if not end:
            if self.drop_off:  # Add vehicle back into vacancy dictionary
                if self.vehicle.is_HV:
                    HVs[self.vehicle.id] = self.vehicle
                else:
                    AVs[self.vehicle.id] = self.vehicle

            if self.vehicle.nextCruise is self:
                self.vehicle.time = self.time
                self.vehicle.loc = self.vehicle.destination
                nextTime = self.vehicle.cruise()  # Execute cruising
                self.vehicle.nextCruise = CruiseTrip(nextTime, self.vehicle)
