import time
import heapq

from Configuration import configs
from Basics import eventQueue
from Control import write_results
from Supply import load_vehicles, HVs, activeAVs
from Demand import load_passengers, NewPassenger, UpdatePhi
from Management import schedule_assignment, manage_AVs


_t0 = time.time()

# Load vehicles into Events
# - HVs are randomly located, initiated within a time span (e.g. 10 min)
# - AVs are inactive at pre-defined depots, with an active initial fleet at time 0
load_vehicles()

# Load passengers into Events
simulationEndTime = load_passengers()

# Schedule assignments into Events
schedule_assignment(simulationEndTime)
print('Last passenger spawns at {} sec.'.format(simulationEndTime))

# manage_AVs()  # TODO: Finish AV control optimisation

# Execute event queue, sorted by Time and Priority
while len(eventQueue) != 0:
    e = heapq.heappop(eventQueue)

    if isinstance(e, UpdatePhi):
        e.trigger(len(HVs), len(activeAVs))
    elif isinstance(e, NewPassenger):
        e.trigger(HVs.values(), activeAVs.values())
    else:
        e.trigger()

# TODO: Account for incomplete events
# 1. Non-expired and Non-assigned passengers

# Output relevant results
# write_results(configs['data_output_path'], configs['output_number'])
print('Simulation ended in: {:4d} sec.'.format(int(time.time() - _t0)))
