import pandas as pd
import numpy as np
import geopandas as gpd
import momepy
import networkx as nx
import matplotlib.pyplot as plt

from Parser import map_file, path_time_file, charging_station_file, nearest_station_file


# Construct Networkx directed graph based on parsed edge list
network = gpd.read_file(map_file)
network = network.to_crs('epsg:2263')  # Manhattan EPSG
G = momepy.gdf_to_nx(network, multigraph=False, directed=True, length='distance')
nx.set_node_attributes(G, {n: n for n in G.nodes}, 'pos')
nx.relabel_nodes(G, {k[1]: v for k, v in nx.get_edge_attributes(G, 'to_node').items()}, False)

# Load the pre-computed shortest path travel times between all intersection OD pairs
shortest_times = pd.read_csv(path_time_file, index_col=0)
shortest_times.columns = shortest_times.columns.astype(np.int64)

# Load the pre-defined charging stations and their allocated areas (intersections)
charging_stations = pd.read_csv(charging_station_file).squeeze()
nearest_station = pd.read_csv(nearest_station_file, index_col=0)


def plot_stations():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600, tight_layout=True)
    ax.set_aspect('equal')
    nx.draw(G, nx.get_node_attributes(G, 'pos'), arrows=False, node_size=0, edge_color='grey')
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), charging_stations,
                           node_size=5, node_shape='^', node_color='b', label='Stations')
    plt.show()
