import pandas as pd

from Configuration import configs
from Basics import np, lonlat_to_loc, distance_between, duration_between


def write_passengers():
    cols = pd.read_csv(configs["passenger_file"], nrows=1).columns
    if 'origin_loc_source' not in cols:
        print('Writing passengers...')
        passenger_df = pd.read_csv(configs["passenger_file"])
        passenger_df['tpep_pickup_datetime'] = (pd.to_datetime(passenger_df['tpep_pickup_datetime']) -
                                                pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        for idx, p in passenger_df.iterrows():
            o = lonlat_to_loc(p['pickup_longitude'], p['pickup_latitude'])
            d = lonlat_to_loc(p['dropoff_longitude'], p['dropoff_latitude'])

            passenger_df.loc[idx, 'origin_loc_source'] = o.source
            passenger_df.loc[idx, 'origin_loc_target'] = o.target
            passenger_df.loc[idx, 'origin_loc_distance'] = np.max([0, o.locFromSource])
            passenger_df.loc[idx, 'destination_loc_source'] = d.source
            passenger_df.loc[idx, 'destination_loc_target'] = d.target
            passenger_df.loc[idx, 'destination_loc_distance'] = np.max([0, d.locFromSource])

            passenger_df.loc[idx, 'trip_distance'] = distance_between(o, d)
            passenger_df.loc[idx, 'trip_duration'] = duration_between(o, d)

        passenger_df.to_csv(configs["passenger_file"], index=False)
        print('Writing is completed.')
    else:
        pass


def read_passengers():
    write_passengers()
    use_cols = ['tpep_pickup_datetime',
                'origin_loc_source', 'origin_loc_target', 'origin_loc_distance',
                'destination_loc_source', 'destination_loc_target', 'destination_loc_distance',
                'trip_distance', 'trip_duration']
    passenger_df = pd.read_csv(configs["passenger_file"], usecols=use_cols)
    passenger_df['time'] = passenger_df['tpep_pickup_datetime']
    passenger_df['time'] -= passenger_df['time'].min()

    return passenger_df.drop(columns=['tpep_pickup_datetime'])
