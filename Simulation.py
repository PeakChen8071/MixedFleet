import time
import heapq

from Configuration import configs
from Basics import eventQueue, validate_passengers
from Control import write_results
from Supply import load_vehicles, HVs, activeAVs, DeactivateAVs, TripCompletion
from Demand import load_passengers, NewPassenger, UpdatePhi, Passenger
from Management import schedule_assignment


_t0 = time.time()

# Load vehicles into Events
# - HVs are randomly located, join the market based on their (1) neoclassical (2) income-targeting behaviours
# - AVs are inactive at pre-defined depots, with an active initial fleet at 04:00
load_vehicles()

# Load passengers into Events
validate_passengers(configs["passenger_file"])
simulationEndTime = load_passengers(1/4)

# Schedule assignments into Events
schedule_assignment(simulationEndTime)
print('Last passenger spawns at {} sec.'.format(simulationEndTime))

while len(eventQueue) != 0:
    event = heapq.heappop(eventQueue)

    if event.time <= simulationEndTime:
        # Execute event queue, sorted by Time and Priority
        if isinstance(event, UpdatePhi):
            event.trigger(len(HVs), len(activeAVs))
        elif isinstance(event, NewPassenger):
            event.trigger(HVs.values(), activeAVs.values())
        else:
            event.trigger()
    else:  # Clear remaining passengers and vehicles
        assert isinstance(event, TripCompletion), 'Overdue events should be TripCompletion'
        event.trigger(end=True)

# Remaining passengers will not be assigned, and expire
for p in (Passenger.p_HV | Passenger.p_AV).values():
    p.check_expiration(999999)

# Deactivate all remaining (active) AVs
DeactivateAVs(simulationEndTime, len(activeAVs))

# Output relevant results
write_results(configs['data_output_path'], configs['output_number'])
print('Simulation ended in: {:4d} sec.'.format(int(time.time() - _t0)))
