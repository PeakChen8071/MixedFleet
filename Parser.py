import pandas as pd

from Configuration import configs


depot_nodes = configs['depot_nodes']


def read_edgeList():
    edgeList = pd.read_csv(configs['map_file'])
    edgeList.set_index('from_node', drop=False, inplace=True)
    edgeList['pos'] = tuple(zip(edgeList['from_node_lon'], edgeList['from_node_lat']))
    return edgeList, 'from_node', 'to_node', ['length', 'travel_time']


def read_passengers(fraction, rows):
    use_cols = ['tpep_pickup_datetime',
                'origin_loc_source', 'origin_loc_target', 'origin_loc_distance',
                'destination_loc_source', 'destination_loc_target', 'destination_loc_distance',
                'trip_distance', 'trip_duration', 'patience', 'VoT']
    passenger_df = pd.read_csv(configs["passenger_file"], usecols=use_cols, nrows=rows)
    passenger_df['time'] = passenger_df['tpep_pickup_datetime']
    passenger_df['time'] -= passenger_df['time'].min()

    # print(passenger_df.dtypes)  # Print data types for debugging

    return passenger_df.drop(columns=['tpep_pickup_datetime']).sample(frac=fraction)
