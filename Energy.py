from Map import charging_stations
from Basics import Event, Location, p_id

unavailable_cs = {}
available_cs = {}


class Electricity:
    max_SoC = 40  # unit: kWh
    # min_SoC = 8  # unit: kWh
    healthy_SoC = 25  # unit: kWh

    consumption_rate = 6  # unit: kW
    price = 0.2  # unit: $/kWh


def get_SoC_value(vehicle_SoC):
    return Electricity.price * (vehicle_SoC - Electricity.healthy_SoC)


class Charger:
    def __init__(self, loc, charge_rate=6, charge_price=1.5):
        self.id = next(p_id)
        self.loc = loc
        self.charge_rate = charge_rate  # unit: kW
        self.charge_price = charge_price  # unit: $/kWh

        available_cs[self.id] = self

    def __repr__(self):
        return 'Charger{}'.format(self.id)

    def charge_time(self, vehicle_SoC):
        return int((Electricity.max_SoC - vehicle_SoC) / self.charge_rate * 3600)

    def charge_cost(self, vehicle_SoC, discount=0):
        return round((1 - discount) * self.charge_price * (Electricity.max_SoC - vehicle_SoC), 2)


# Load static chargers into the network
for cs in charging_stations:
    Charger(Location(cs))


def get_charge_benefit(vehicle_SoC):
    return 50 * 0.9 ** vehicle_SoC
    # return (1 - vehicle_SoC / Electricity.max_SoC) * 50


class ChargerOn(Event):
    def __init__(self, time, charger_id):
        super().__init__(time, priority=1)
        self.charger_id = charger_id

    def __repr__(self):
        return 'Charger{}On@t{}'.format(self.charger_id, self.time)

    def trigger(self):
        available_cs[self.charger_id] = unavailable_cs.pop(self.charger_id)
