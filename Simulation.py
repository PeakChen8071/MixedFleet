import time
import heapq

from Configuration import configs
from Basics import eventQueue, validate_passengers
from Control import Statistics, Variables, write_results, MPC
from Supply import load_vehicles, load_simple_vehicles, HVs, activeAVs, DeactivateAVs, TripCompletion
from Demand import load_passengers, NewPassenger, UpdatePhi, Passenger
from Management import schedule_states, schedule_assignment, schedule_MPC, manage_AVs


_t0 = time.time()

# Load passengers into Events
validate_passengers(configs['passenger_file'])
load_passengers()
print('Last passenger spawns at {} sec.'.format(Statistics.lastPassengerTime))

# Load vehicles into Events
# - HVs are randomly located, join the market based on their (1) neoclassical (2) income-targeting behaviours
# - AVs are inactive at pre-defined depots, with an active initial fleet at 04:00
load_vehicles(neoclassical=0)
# load_simple_vehicles()

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
            manage_AVs(event.time+1, Variables.AV_change)
            Variables.AV_change = 0
        else:
            event.trigger()
    else:  # Clear remaining passengers and vehicles
        if len(HVs) != 0:
            for _v in HVs.values():  # All vacant HVs force exit the market
                _v.decide_exit(event.time, end=True)
            HVs.clear()

        assert isinstance(event, TripCompletion), 'Overdue events should not be {}'.format(event)
        event.trigger(end=True)  # All occupied HVs force exit the market after drop-off
        Statistics.simulationEndTime = event.time

# Deactivate all remaining (active) AVs
DeactivateAVs(Statistics.simulationEndTime, len(activeAVs)).trigger()

# Remaining passengers will not be assigned, and cancel their orders
for p in (Passenger.p_HV | Passenger.p_AV).values():
    p.check_expiration(999999)

# Output relevant results
write_results(configs['data_output_path'], configs['output_number'])
print('Simulation wall time: {:5d} sec.'.format(int(time.time() - _t0)))

