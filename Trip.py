import pandas as pd
import pulp

from Parser import match_interval
from Map import nearest_station
from Basics import Event, duration_between
from Control import Parameters, Variables, Statistics
from Energy import Electricity, Charger, available_cs, unavailable_cs, ChargerOn, get_SoC_value, get_charge_benefit
from Demand import passengers, Passenger
from Supply import EVs, TripCompletion


class Assign(Event):
    def __init__(self, time):
        super().__init__(time, priority=4)

    def __repr__(self):
        return 'Assignment@t{}'.format(self.time)

    def trigger(self, end=False):
        if not end:
            for v in EVs.copy().values():
                v.time = self.time  # Update vehicle time
                v.decide_exit(self.time)  # Each vacant EV decides whether to exit market

            for p in passengers.copy().values():
                p.check_expiration(self.time)  # Remove expired passengers

            # Compute and return minimum weighting full bipartite matching
            matches = match_trips(EVs.values(), passengers.values())
            print(matches)

            # TODO: finialise transport method
            self.transport(matches)

    def transport(self, match_results):
        if match_results:
            for trip in match_results:
                vehicle = EVs.pop(trip[0].id)  # Remove vehicle from available dictionary
                target = trip[1]  # Assigned passenger or station or location
                meeting_t = self.time + trip[2]  # Timestamp of meeting
                vehicle.SoC -= trip[2] / 3600 * Electricity.consumption_rate  # Meeting SoC is consumed

                if isinstance(target, Passenger):  # Vehicle assigned to a passenger
                    # Remove passenger from waiting list
                    del passengers[target.id]

                    # Vehicle delivers passenger to passenger destination
                    delivery_t = meeting_t + target.tripDuration  # Timestamp of drop-off
                    vehicle.time = delivery_t
                    vehicle.loc = target.destination
                    vehicle.income += Parameters.wage * target.tripDuration / 3600  # EV receives trip income
                    vehicle.SoC -= target.tripDuration / 3600 * Electricity.consumption_rate  # Trip SoC is consumed

                    # Update system statistics
                    vehicle.occupiedTime = target.tripDuration  # Update vehicle occupied duration
                    PickUp(meeting_t)

                    # Record trip statistics
                    Statistics.data_output['assignment_data'].append([vehicle.id, target.id, False, self.time, meeting_t, delivery_t, trip[3], vehicle.SoC])

                    # Their next trip planning occurs after passenger drop-off
                    vehicle.nextTrip = TripCompletion(delivery_t, vehicle, drop_off=True)

                elif isinstance(target, Charger):
                    # Remove charger from available list
                    unavailable_cs[target.id] = available_cs.pop(target.id)

                    # Vehicle recharges at the charging station
                    complete_t = meeting_t + target.charge_time(vehicle.SoC)
                    vehicle.time = complete_t
                    vehicle.loc = target.loc
                    vehicle.income -= target.charge_cost(vehicle.SoC)
                    vehicle.SoC = Electricity.max_SoC  # Recharge to full SoC

                    # Record trip statistics
                    Statistics.data_output['assignment_data'].append([vehicle.id, target.id, True, self.time, meeting_t, complete_t, trip[3], vehicle.SoC])

                    # Their next trip planning occurs after recharging
                    ChargerOn(complete_t, target.id)
                    vehicle.nextTrip = TripCompletion(complete_t, vehicle, drop_off=False)
                else:
                    raise ValueError('Vehicle is not matched to a passenger or charging station')


def match(Utility, constr_list):
    # Define LP problem
    model = pulp.LpProblem("Ride_Matching_Problems", pulp.LpMaximize)

    # Initialise binary variables to represent pairing
    X = pulp.LpVariable.dicts("X", ((_v, _p) for _v in Utility.index for _p in Utility.columns), lowBound=0, upBound=1, cat='Integer')
    for O, D in constr_list:
        X[O, D].setInitialValue(0)
        X[O, D].fixValue()

    # Objective as the sum of potential costs
    model += (pulp.lpSum([Utility.loc[_v, _p] * X[(_v, _p)] for _v in Utility.index for _p in Utility.columns]))

    # Constraint 1: row sum to 1, one vacant vehicle only matches with one waiting passenger request
    for _v in Utility.index:
        model += pulp.lpSum([X[(_v, _p)] for _p in Utility.columns]) <= 1

    for _p in Utility.columns:
        model += pulp.lpSum([X[(_v, _p)] for _v in Utility.index]) <= 1

    solver = pulp.COIN_CMD(msg=False)  # Suppress output
    model.solve(solver)  # Solve LP

    result = {}
    for var in X:
        var_value = X[var].varValue
        if var_value != 0:
            # Write paired matches to result dict
            result[var[0]] = var[1]

    return result


def match_trips(vacant_v, waiting_p):
    profit_matrix = pd.DataFrame(index=vacant_v, columns=list(waiting_p) + list(available_cs.values()))

    vehicle_SoC = {v: get_SoC_value(v.SoC) for v in vacant_v}

    passenger_fare = {p: p.fare for p in waiting_p}
    passenger_wage = {p: round(Parameters.wage * p.tripDuration / 3600, 2) for p in waiting_p}
    trip_tt = pd.DataFrame(index=vacant_v, columns=list(waiting_p) + list(available_cs.values()))

    constraints = []

    for v in vacant_v:
        for p in waiting_p:
            _tt = duration_between(v.loc, p.origin)
            trip_tt.loc[v, p] = _tt

            if v.SoC * 3600 / Electricity.consumption_rate < _tt + p.tripDuration + nearest_station.loc[p.destination.target, 'tt']:
                constraints.append((v, p))
                profit_matrix.loc[v, p] = -1e6
            else:
                profit_matrix.loc[v, p] = passenger_fare[p] - passenger_wage[p] \
                                          - 0.01 * _tt \
                                          - Electricity.price * Electricity.consumption_rate * (_tt + p.tripDuration) / 3600 \
                                          + vehicle_SoC[v]

        for cs in available_cs.values():
            _tt = duration_between(v.loc, cs.loc)
            trip_tt.loc[v, cs] = _tt

            if v.SoC * 3600 / Electricity.consumption_rate < _tt:
                constraints.append((v, cs))
                profit_matrix.loc[v, cs] = -1e6
            else:
                profit_matrix.loc[v, cs] = get_charge_benefit(v.SoC) - cs.charge_cost(v.SoC)  \
                                           - 0.01 * _tt \
                                           - Electricity.price * Electricity.consumption_rate * _tt / 3600 \
                                           - vehicle_SoC[v]

    results = [(v, p, trip_tt.loc[v, p], profit_matrix.loc[v, p]) for v, p in match(profit_matrix, constraints).items()]
    return results  # List of tuples of assigned (vehicle, passenger, trip_tt)


# Schedule assignment events, finish with assignment to catch all passengers
def schedule_assignment(last_passenger_time):
    for t in range(0, last_passenger_time + match_interval, match_interval):
        Assign(t)
        Statistics.simulationEndTime = t


class PickUp(Event):
    def __init__(self, time):
        super().__init__(time, priority=1)

    def __repr__(self):
        return 'Pick-up@t{}'.format(self.time)

    def trigger(self):
        Variables.EV_no += 1
