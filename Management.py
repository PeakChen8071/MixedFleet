import itertools

import pandas as pd
from gurobipy import *

from Configuration import configs
from Basics import Event, duration_between_vec
from Control import Parameters, Variables, Statistics, MPC
from Demand import Passenger
from Supply import HVs, activeAVs, TripCompletion


def bipartite_match(vacant_v, waiting_p):
    if (not vacant_v) or (not waiting_p):
        return None  # No matching if either set is empty

    trip_tt = pd.DataFrame((duration_between_vec([v.loc for v in vacant_v], [p.origin for p in waiting_p])).reshape(len(vacant_v), len(waiting_p)), index=vacant_v, columns=waiting_p)
    Utility = trip_tt.rdiv(1, fill_value=0)
    utility_dict = Utility.stack().to_dict()

    # Define LP problem
    model = Model('bipartite_matching')
    model.setParam('OutputFlag', 0)
    X = model.addVars(list(itertools.product(Utility.index, Utility.columns)), vtype=GRB.BINARY, name='x')
    model.setObjective(X.prod(utility_dict), GRB.MAXIMIZE)

    # Objective as the sum of potential costs
    model.addConstrs((X.sum('*', _p) <= 1 for _p in Utility.columns))
    model.addConstrs((X.sum(_v, '*') <= 1 for _v in Utility.index))

    results = []
    model.optimize()

    for key in X:
        if X[key].x == 1:
            results.append((key[0], key[1], trip_tt.loc[key]))

    return results  # List of tuples of assigned (vehicle, passenger, trip_tt)


class Assign(Event):
    def __init__(self, time):
        super().__init__(time, priority=5)

    def __lt__(self, other):
        return (self.time, self.priority) < (other.time, other.priority)

    def __repr__(self):
        return 'Assignment@t{}'.format(self.time)

    def transport(self, match_results):
        if match_results:
            for m in match_results:
                v = m[0]  # Assigned vehicle
                p = m[1]  # Assigned passenger
                meeting_t = self.time + m[2]  # Timestamp of meeting
                delivery_t = meeting_t + p.tripDuration  # Timestamp of drop-off
                v.occupiedTime = p.tripDuration  # Update vehicle occupied duration

                # Vehicle and passenger are assigned and removed from available dictionaries
                if v.is_HV:
                    del HVs[v.id]
                    del Passenger.p_HV[p.id]

                    # Update HV meeting time expectation
                    if self.time > 3600:
                        Variables.HV_ta = (Variables.HV_ta * Variables.HV_trips + m[2]) / (Variables.HV_trips + 1)
                        Variables.HV_to = (Variables.HV_to * Variables.HV_trips + p.tripDuration) / (Variables.HV_trips + 1)

                    # HVs receive incomes (wage = unit wage * trip duration)
                    v.income += Variables.HV_wage / 3600 * p.tripDuration
                    Variables.totalWage += Variables.HV_wage / 3600 * p.tripDuration

                    # Add trip pick-up / drop-off feedback
                    if meeting_t in Statistics.HV_pickup_counter:
                        Statistics.HV_pickup_counter[meeting_t] += 1
                    else:
                        Statistics.HV_pickup_counter[meeting_t] = 1

                    if delivery_t in Statistics.HV_dropoff_counter:
                        Statistics.HV_dropoff_counter[delivery_t] += 1
                    else:
                        Statistics.HV_dropoff_counter[delivery_t] = 1

                else:
                    del activeAVs[v.id]
                    del Passenger.p_AV[p.id]

                    if self.time > 3600:
                        Variables.AV_ta = (Variables.AV_ta * Variables.AV_trips + m[2]) / (Variables.AV_trips + 1)
                        Variables.AV_to = (Variables.AV_to * Variables.AV_trips + p.tripDuration) / (Variables.AV_trips + 1)

                    if meeting_t in Statistics.AV_pickup_counter:
                        Statistics.AV_pickup_counter[meeting_t] += 1
                    else:
                        Statistics.AV_pickup_counter[meeting_t] = 1

                    if delivery_t in Statistics.AV_dropoff_counter:
                        Statistics.AV_dropoff_counter[delivery_t] += 1
                    else:
                        Statistics.AV_dropoff_counter[delivery_t] = 1

                # Vehicle delivers passenger to passenger destination
                v.time = delivery_t
                v.loc = p.destination

                # Their next trip planning occurs after delivery trip completion
                v.nextTrip = TripCompletion(delivery_t, v, drop_off=True)

                # Update system statistics
                if meeting_t < Statistics.simulationEndTime:
                    UpdateOccupied(meeting_t, v.is_HV, 1)
                if delivery_t < Statistics.simulationEndTime:
                    UpdateOccupied(delivery_t, v.is_HV, -1)

                # Record trip statistics
                Statistics.assignment_data.append([v.id, p.id, v.is_HV, self.time, meeting_t, delivery_t])

    def trigger(self, end=False):
        if self.time % 3600 == 0:
            print('Hour = {:d}:00; AV fleet = {}; HV fleet = {}; AV trips = {}; HV trips = {}'.format(
                4 + self.time // 3600, Variables.AV_total, Variables.HV_total, Variables.AV_trips, Variables.HV_trips))

        if not end:
            for v in (HVs | activeAVs).values():
                v.time = self.time  # Update vehicle time

            for p in (Passenger.p_HV | Passenger.p_AV).values():
                p.check_expiration(self.time)  # Remove expired passengers

            # Compute and return minimum weighting full bipartite matching
            HV_match = bipartite_match(HVs.values(), Passenger.p_HV.values())
            AV_match = bipartite_match(activeAVs.values(), Passenger.p_AV.values())

            self.transport(HV_match)
            self.transport(AV_match)

        Variables.AV_nv = len(activeAVs)
        Variables.HV_nv = len(HVs)
        Variables.HV_pw = len(Passenger.p_HV)
        Variables.AV_pw = len(Passenger.p_AV)


class UpdateStates(Event):
    def __init__(self, time):
        super().__init__(time, priority=4)

    def __repr__(self):
        return 'UpdateStates@t{}'.format(self.time)

    def trigger(self):
        Variables.AV_pw = len(Passenger.p_AV)
        Variables.AV_nv = len(activeAVs)
        Variables.AV_na = Variables.AV_total - Variables.AV_nv - Variables.AV_no
        Variables.HV_pw = len(Passenger.p_HV)
        Variables.HV_nv = len(HVs)
        Variables.HV_na = Variables.HV_total - Variables.HV_nv - Variables.HV_no

        if (Variables.AV_total > 0) and (self.time > 3600):
            Parameters.AV_occupancy = Variables.AV_no / Variables.AV_total

        if (Variables.HV_total > 0) and (self.time > 3600):
            Parameters.HV_occupancy = Variables.HV_no / Variables.HV_total


class OverwriteControls(Event):
    def __init__(self, time, AV_fare=None, HV_fare=None, AV_change=None):
        super().__init__(time, priority=4)
        self.AV_fare = AV_fare
        self.HV_fare = HV_fare
        self.AV_change = AV_change

    def __repr__(self):
        return 'OverwriteControls@t{}'.format(self.time)

    def trigger(self):
        if self.AV_fare:
            Variables.AV_unitFare = self.AV_fare
        if self.HV_fare:
            Variables.HV_unitFare = self.HV_fare
        if self.AV_change:
            Variables.AV_change = self.AV_change


def schedule_states(endTime):
    for t in range(endTime):
        UpdateStates(t)


# Schedule assignment events, finish with assignment to catch all passengers
def schedule_assignment(endTime):
    roundEndTime = 0
    for t in range(0, endTime + configs['match_interval'], configs['match_interval']):
        Assign(t)
        roundEndTime = t
    Statistics.lastPassengerTime = roundEndTime


def schedule_MPC(endTime):
    for t in range(configs['MPC_start_hour'] * 3600, configs['MPC_end_hour'] * 3600, configs['MPC_control_interval']):
        if t < endTime:
            MPC(t, N=configs['MPC_steps'], Nc=configs['MPC_control_steps'],
                tau_c=configs['MPC_control_interval'], tau_k=configs['MPC_prediction_interval'])
        else:
            raise ValueError('MPC cannot be scheduled after simulation ends!')

    # Clear MPC AV fleet control at the end of MPCs
    OverwriteControls(configs['MPC_end_hour'] * 3600, AV_fare=36, HV_fare=36, AV_change=0)


class UpdateOccupied(Event):
    def __init__(self, time, isHV, change):
        super().__init__(time, priority=0)
        self.isHV = isHV
        self.change = change

    def __repr__(self):
        return 'UpdateOccupied@t{} {}'.format(self.time, self.change)

    def trigger(self):
        if self.isHV:
            Variables.HV_no += self.change
        else:
            Variables.AV_no += self.change
