import time
import heapq

from Supply import deploy_HV, deploy_AV, HV, AV
from Demand import simulationEndTime, load_passengers, Passenger
from Interaction import update_phi, compute_assignment, Events

# EventTypes = {0: 'Update_phi', 1: 'Passenger', 2: 'Cruise', 3: 'Assignment'}
_t0 = time.time()

deploy_HV()  # Randomly deploy Vehicles based on Configs
deploy_AV()
load_passengers()  # Load passengers into Events

t = 0
while (t < simulationEndTime + 1) and (len(Events) != 0):
    event = heapq.heappop(Events)
    eventType = event[1]
    t = event[0]

    if eventType == 0:
        update_phi()
    elif eventType == 1:
        Passenger(t, event[3], event[4], event[5], event[6], HV.HV_v.values(), AV.AV_v.values())
    # elif eventType == 2:
    #     vehicle_id = event[2]
    #     if vehicle_id in HV.HV_v.keys():
    #         HV.HV_v[vehicle_id].cruise()
    #     elif vehicle_id in AV.AV_v.keys():
    #         AV.AV_v[vehicle_id].cruise()
    #     else:
    #         raise ValueError('Cruising vehicle is not available.')


# for t in range(1, simulationEndTime+1, 10):
#     heapq.heappush(Events, (t, 2))

print('Time elapsed: {:4d} sec.'.format(int(time.time() - _t0)))
