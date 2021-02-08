import heapq
from itertools import count
import bisect

from Configuration import configs
from Basics import G, random_loc, path_between, Location, Events
from Control import Variables


def deploy_HV():
    for _ in range(configs['HV_fleet_size']):
        HV(0, random_loc())


def deploy_AV():
    for _ in range(configs['AV_fleet_size']):
        AV(0, random_loc())


class Vehicle:
    _ids = count(0)

    def __init__(self, start_time, start_loc):
        self.id = next(self._ids)
        self.time = start_time
        self.loc = start_loc
        self.state = 'vacant'
        self.pathNodes, self.pathTimes = self.cruise('random_destination')

    def cruise(self, plan):
        if plan is 'random_destination':
            startTime = self.time
            destination = random_loc()
            pathTimes = []
            pathNodes = path_between(self.loc, destination)
            if self.loc.type is not 'Intersection':
                startTime += self.loc.timeFromTarget
                pathTimes.append(startTime)
            for i in range(len(pathNodes) - 1):
                startTime += G.edges[pathNodes[i], pathNodes[i + 1]]['travel_time']
                pathTimes.append(startTime)
            if destination.type is not 'Intersection':
                pathTimes.append(startTime + destination.timeFromSource)
            heapq.heappush(Events, (pathTimes[-1], 2, self.id))
        else:
            raise ValueError('Unrecognised cruising plan.')
        return pathNodes, pathTimes

    def update(self, t):
        idx = bisect.bisect_left(self.pathTimes, t)
        if idx == 0:
            deltaT = t - self.time
            deltaD = deltaT * self.loc.length / self.loc.travelTime
            self.loc.timeFromTarget -= deltaT
            self.loc.timeFromSource += deltaT
            self.loc.locFromTarget -= round(deltaD, 2)
            self.loc.locFromSource += round(deltaD, 2)
        else:
            self.pathNodes = self.pathNodes[idx-1::]
            self.pathTimes = self.pathTimes[idx-1::]
            source = self.pathNodes[0]
            target = self.pathNodes[1]
            pos = (t - self.pathTimes[0]) * G.edges[source, target]['length'] / G.edges[source, target]['travel_time']
            self.loc = Location(source, target, round(pos, 2))
        self.time = t


class HV(Vehicle):
    vehicleType = 'HV'
    HV_v = {}
    # HV_a = {}
    HV_o = {}

    def __init__(self, time, loc):
        super().__init__(time, loc)
        HV.HV_v[self.id] = self

    def __repr__(self):
        return 'HV_{}'.format(self.id)

    def exit(self):
        pass


class AV(Vehicle):
    vehicleType = 'AV'
    AV_v = {}
    # AV_a = {}
    AV_o = {}

    def __init__(self, loc, time):
        super().__init__(loc, time)
        AV.AV_v[self.id] = self

    def __repr__(self):
        return 'AV_{}'.format(self.id)

    def activate(self):
        pass

    def deactivate(self):
        # Go to nearest parking facility for charging and maintenance
        pass
