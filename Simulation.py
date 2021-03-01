import time
import heapq

from Configuration import configs
from Basics import eventQueue
from Control import write_results
from Supply import load_vehicles, HVs, AVs, parkedAVs, CruiseTrip
from Demand import load_passengers, NewPassenger, UpdatePhi
from Interaction import Assign

_t0 = time.time()

load_vehicles()  # Randomly deploy Vehicles based on Configs
simulationEndTime = load_passengers(1/6)  # Load passengers into Events
print('Last passenger at {} sec.'.format(simulationEndTime))

for t in range(0, simulationEndTime + configs['match_interval'], configs['match_interval']):
    Assign(t)  # Schedule assignment events, finish with assignment to catch all passengers

# Execute event queue, sorted by Time and Priority
while len(eventQueue) != 0:
    e = heapq.heappop(eventQueue)

    if isinstance(e, CruiseTrip) and e.time > simulationEndTime:
        e.trigger(end=True)
    elif isinstance(e, UpdatePhi):
        e.trigger(len(HVs), len(AVs))
    elif isinstance(e, NewPassenger):
        e.trigger(HVs.values(), AVs.values())
    else:
        e.trigger()

    # # Change AV numbers
    # if (e.time >= 3600) and (e.time < 7200):
    #     while len(AVs) > 50:
    #         for v in AVs.values():
    #             v.deactivate()
    # elif e.time >= 7200:
    #     for v in parkedAVs:
    #         v.activate()

# TODO: Account for incomplete events
# 1. Non-expired and Non-assigned passengers

# # Output relevant results
# write_results(configs['data_output_path'], configs['output_number'])


print('Simulation ended in: {:4d} sec.'.format(int(time.time() - _t0)))
