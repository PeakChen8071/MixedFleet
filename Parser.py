import pandas as pd

from Configuration import configs


depot_nodes = configs['depot_nodes']
map_file = configs['map_file']
path_time_file = configs['shortest_path_time_file']


def read_passengers(fraction, hours):
    use_cols = ['tpep_pickup_datetime', 'o_source', 'o_target', 'o_loc', 'd_source', 'd_target', 'd_loc',
                'trip_distance', 'trip_duration', 'patience', 'VoT']
    passenger_df = pd.read_csv(configs["passenger_file"], usecols=use_cols)
    passenger_df['time'] = passenger_df['tpep_pickup_datetime'] - passenger_df['tpep_pickup_datetime'].min()

    # Demand time shift, move 00:00 - 04:00 to the end of the day such that simulation starts at 04:00
    passenger_df['time'] = (passenger_df['time'] + 20 * 3600) % (24 * 3600)

    # Limit daily demand to the specified hours
    passenger_df = passenger_df[passenger_df['time'] <= hours * 3600]

    # print(passenger_df.dtypes)  # Print data types for debugging

    return passenger_df.drop(columns=['tpep_pickup_datetime']).sample(frac=fraction).reset_index()
