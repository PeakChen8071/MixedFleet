import json

with open('Config.json', 'r') as jsonFile:
    configs = json.load(jsonFile)


maximum_work = configs['maximum_work_duration']
map_file = configs['map_file']
path_time_file = configs['shortest_path_time_file']
charging_station_file = configs['charging_station_file']
nearest_station_file = configs['nearest_station_file']
passenger_file = configs["passenger_file"]

match_interval = configs['match_interval']
default_waiting_time = configs['default_waiting_time']
fleet_size = configs['fleet_size']

output_path = configs['output_path']
output_number = configs['output_number']
