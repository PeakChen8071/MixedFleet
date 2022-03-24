import pandas as pd
import pulp

from Configuration import configs
from Basics import Event, duration_between
from Control import Parameters, Variables, Statistics, MPC
from Demand import Passenger
from Supply import HVs, activeAVs, TripCompletion, ActivateAVs, DeactivateAVs


def One2One_Matching(Utility):
    # Define LP problem
    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    # Initialise binary variables to represent pairing
    X = pulp.LpVariable.dicts("X", ((_v, _p) for _v in Utility.index for _p in Utility.columns), lowBound=0, upBound=1, cat='Integer')

    # Objective as the sum of potential costs
    model += (pulp.lpSum([Utility.loc[_v, _p] * X[(_v, _p)] for _v in Utility.index for _p in Utility.columns]))

    # Constraint 1: row sum to 1, one vacant vehicle only matches with one waiting passenger request
    for _v in Utility.index:
        model += pulp.lpSum([X[(_v, _p)] for _p in Utility.columns]) <= 1

    for _p in Utility.columns:
        model += pulp.lpSum([X[(_v, _p)] for _v in Utility.index]) <= 1

    solver = pulp.GLPK_CMD(msg=False)  # Suppress output
    model.solve(solver)  # Solve LP

    result = {}
    for var in X:
        var_value = X[var].varValue
        if var_value != 0:
            # Write paired matches to result dict
            result[var[0]] = var[1]

    return result


# Bipartite matching which minimises the total dispatch trip duration
# def bipartite_match(vacant_v, waiting_p):
#     if (not vacant_v) | (not waiting_p):
#         return None  # No matching if either set is empty
#
#     bipartite_graph = nx.DiGraph()
#     for p in waiting_p:
#         for v in vacant_v:
#             # _t = duration_between(v.loc, p.origin)
#             bipartite_graph.add_node(v, bipartite=0)
#             bipartite_graph.add_node(p, bipartite=1)
#             bipartite_graph.add_edge(v, p, duration=duration_between(v.loc, p.origin))
#
#     results = None
#     top_nodes = {n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 0}
#     if not nx.is_empty(bipartite_graph):
#         results = [(v, p, bipartite_graph.edges[v, p]['duration'])
#                    for v, p in nx.bipartite.minimum_weight_full_matching(bipartite_graph, top_nodes=top_nodes, weight='duration').items()
#                    if isinstance(p, Passenger)]
#
#     return results  # List of tuples with assigned (vehicle, passenger)


def bipartite_match(vacant_v, waiting_p):
    if (not vacant_v) or (not waiting_p):
        return None  # No matching if either set is empty

    trip_tt = pd.DataFrame(index=vacant_v, columns=waiting_p)
    for v in vacant_v:
        for p in waiting_p:
            trip_tt.loc[v, p] = duration_between(v.loc, p.origin)

    results = [(v, p, trip_tt.loc[v, p]) for v, p in One2One_Matching(trip_tt.replace(0, 1).rdiv(1)).items()]

    return results  # List of tuples of assigned (vehicle, passenger, trip_tt)


def compute_assignment(t):
    for v in (HVs | activeAVs).values():
        # v.update_loc(t)  # Update vehicle location
        v.time = t  # Update vehicle time

    for p in (Passenger.p_HV | Passenger.p_AV).values():
        p.check_expiration(t)  # Remove expired passengers

    # Compute and return minimum weighting full bipartite matching
    HV_match = bipartite_match(HVs.values(), Passenger.p_HV.values())
    AV_match = bipartite_match(activeAVs.values(), Passenger.p_AV.values())
    return HV_match, AV_match


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
                Statistics.assignment_data = Statistics.assignment_data.append({'v_id': v.id,
                                                                                'p_id': p.id,
                                                                                'is_HV': v.is_HV,
                                                                                'dispatch_t': self.time,
                                                                                'meeting_t': meeting_t,
                                                                                'delivery_t': delivery_t},
                                                                               ignore_index=True)

    def trigger(self, end=False):
        if not end:
            HV_match, AV_match = compute_assignment(self.time)
            self.transport(HV_match)
            self.transport(AV_match)

            # Exit decision-making at matching intervals for vacant HVs
            for k in HVs.copy().keys():
                HVs.pop(k).decide_exit(self.time)

            Variables.histExits.append(Variables.exitDecisions)
            Variables.exitDecisions = 0


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

        if Variables.AV_total > 0:
            Parameters.AV_occupancy = Variables.AV_no / Variables.AV_total

        if Variables.HV_total > 0:
            Parameters.HV_occupancy = Variables.HV_no / Variables.HV_total


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


# Dynamic AV fleet management: activation/deactivation
def manage_AVs(time, size):
    if size > 0:
        ActivateAVs(time, size)
    elif size < 0:
        DeactivateAVs(time, -size)
