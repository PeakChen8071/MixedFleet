import pandas as pd

from Configuration import configs
from Basics import np, lonlat_to_loc, distance_between, duration_between


def write_passengers():
    cols = pd.read_csv(configs["passenger_file"], nrows=1).columns
    if 'origin_loc_source' not in cols:
        print('Writing passengers...')
        df = pd.read_csv(configs["passenger_file"])
        df['tpep_pickup_datetime'] = (pd.to_datetime(df['tpep_pickup_datetime']) -
                                      pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        df['Location_o'] = df.apply(lambda x: lonlat_to_loc(x['pickup_longitude'], x['pickup_latitude']), axis=1)
        df['Location_d'] = df.apply(lambda x: lonlat_to_loc(x['dropoff_longitude'], x['dropoff_latitude']), axis=1)

        df['origin_loc_source'] = [o.source for o in df['Location_o']]
        df['origin_loc_target'] = [o.target for o in df['Location_o']]
        df['origin_loc_distance'] = [np.max([0, o.locFromSource]) for o in df['Location_o']]

        df['destination_loc_source'] = [d.source for d in df['Location_d']]
        df['destination_loc_target'] = [d.target for d in df['Location_d']]
        df['destination_loc_distance'] = [np.max([0, d.locFromSource]) for d in df['Location_d']]

        df['trip_distance'] = df.apply(lambda x: distance_between(x['Location_o'], x['Location_d']), axis=1)
        df['trip_duration'] = df.apply(lambda x: duration_between(x['Location_o'], x['Location_d']), axis=1)

        df.drop(columns=['Location_o', 'Location_d']).sort_values(
            'tpep_pickup_datetime').to_csv(configs["passenger_file"], index=False)
        print('Writing is completed.')


def read_passengers(fraction, rows):
    write_passengers()
    use_cols = ['tpep_pickup_datetime',
                'origin_loc_source', 'origin_loc_target', 'origin_loc_distance',
                'destination_loc_source', 'destination_loc_target', 'destination_loc_distance',
                'trip_distance', 'trip_duration']
    passenger_df = pd.read_csv(configs["passenger_file"], usecols=use_cols, nrows=rows)
    passenger_df['time'] = passenger_df['tpep_pickup_datetime']
    passenger_df['time'] -= passenger_df['time'].min()

    return passenger_df.drop(columns=['tpep_pickup_datetime']).sample(frac=fraction)
