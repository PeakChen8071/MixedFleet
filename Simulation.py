import time
import heapq

from Basics import eventQueue
from Control import Statistics, write_results
from Supply import load_simple_vehicles, EVs
from Demand import validate_passengers, load_passengers, NewPassenger, passengers
from Trip import schedule_assignment


_t0 = time.time()


# Load passengers into Events
validate_passengers()
load_passengers(0.05)
print('Last passenger spawns at {} sec.'.format(Statistics.lastPassengerTime))

# Load vehicles into Events
# - EVs are randomly located, join the market based on their (1) neoclassical (2) income-targeting behaviours
# load_vehicles(neoclassical=0)
load_simple_vehicles()
#
# # Schedule assignments into Events
schedule_assignment(Statistics.lastPassengerTime)

while len(eventQueue) != 0:
    event = heapq.heappop(eventQueue)

    if event.time <= Statistics.lastPassengerTime:
        # Execute event queue, sorted by Time and Priority
        if isinstance(event, NewPassenger):
            event.trigger(EVs.values())
        else:
            event.trigger()
    else:  # Clear remaining passengers and vehicles
        for _v in EVs.values():  # All vacant EVs force exit the market
            _v.decide_exit(event.time, end=True)

        # assert isinstance(event, TripCompletion), 'Overdue events should not be {}'.format(event)
        event.trigger(end=True)  # All occupied EVs force exit the market after drop-off
        Statistics.simulationEndTime = event.time


# Remaining passengers will not be assigned, and cancel their orders
for p in passengers.copy().values():
    p.check_expiration(999999)

# Output relevant results
write_results()
print('Simulation wall time: {:5d} sec.'.format(int(time.time() - _t0)))
