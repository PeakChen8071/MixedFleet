import time
import heapq

from Configuration import configs
from Basics import eventQueue, validate_passengers
from Control import Statistics, Variables, write_results, MPC
from Supply import load_vehicles, HVs, activeAVs, TripCompletion, ManageAVs
from Demand import load_passengers, NewPassenger, UpdatePhi, Passenger
from Management import schedule_states, schedule_assignment, schedule_MPC

_t0 = time.time()

# Load passengers into Events
validate_passengers(configs['passenger_file'])
load_passengers()

print('Output Number: ', configs['output_number'])
print('Last passenger spawns at {} sec.'.format(Statistics.lastPassengerTime))

# Load vehicles into Events
load_vehicles(neoclassical=configs['neoclassical'])

# Schedule assignments into Events
schedule_assignment(Statistics.lastPassengerTime)
schedule_states(Statistics.lastPassengerTime)
schedule_MPC(Statistics.lastPassengerTime)

while len(eventQueue) != 0:
    event = heapq.heappop(eventQueue)

    if event.time <= Statistics.lastPassengerTime:
        # Execute event queue, sorted by Time and Priority
        if isinstance(event, UpdatePhi):
            event.trigger(len(HVs), len(activeAVs))
        elif isinstance(event, NewPassenger):
            event.trigger(HVs.values(), activeAVs.values())
        elif isinstance(event, MPC):
            event.trigger()
            ManageAVs(event.time + 1, Variables.AV_change)  # AV fleet control is delayed by 1 sec due to event priority
        else:
            event.trigger()

        # Apply reactive controls per minute between 08:00 and 20:00
        # if (event.time >= 4 * 3600) and (event.time < 16 * 3600):
        #
        #     if event.time % 10 == 0:  # Reactive fleet control per 10 seconds
        #         if (Variables.AV_nv < 10) and (Variables.AV_total < configs['AV_fleet_size']):
        #             ManageAVs(event.time + 1, 1)
        #         elif Variables.AV_nv > 50:
        #             ManageAVs(event.time + 1, -1)
        #
        #     if event.time % 300 == 0:  # Reactive fare control per 5 min
        #         if Variables.HV_nv > Variables.HV_pw:
        #             Variables.HV_unitFare = max(Variables.HV_unitFare - 1, 30)
        #         else:
        #             Variables.HV_unitFare = min(Variables.HV_unitFare + 1, 180)
        #
        #         if Variables.AV_nv > Variables.AV_pw:
        #             Variables.AV_unitFare = max(Variables.AV_unitFare - 1, 30)
        #         else:
        #             Variables.AV_unitFare = min(Variables.AV_unitFare + 1, 180)

    else:  # Clear remaining passengers and vehicles
        if len(HVs) != 0:
            for _v in HVs.values():  # All vacant HVs force exit the market
                _v.decide_exit(event.time, end=True)
            HVs.clear()

        assert isinstance(event, TripCompletion), 'Overdue events should not be {}'.format(event)
        event.trigger(end=True)  # All occupied HVs force exit the market after drop-off
        Statistics.simulationEndTime = event.time

# Deactivate all remaining (active) AVs
ManageAVs(Statistics.lastPassengerTime, -len(activeAVs)).trigger()

# Remaining passengers will not be assigned, and cancel their orders
for p in (Passenger.p_HV | Passenger.p_AV).values():
    p.check_expiration(999999)

# Output relevant results
write_results(configs['data_output_path'], configs['output_number'])
print('Simulation wall time: {:5d} sec.'.format(int(time.time() - _t0)))
