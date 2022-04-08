import pandas as pd
import numpy as np
import geopandas as gpd
import momepy
import networkx as nx
import matplotlib.pyplot as plt

from Parser import map_file, path_time_file, depots, chargingStations, nearestStations


# Construct Networkx directed graph based on parsed edge list
network = gpd.read_file(map_file)
network = network.to_crs('epsg:2263')  # Manhattan EPSG
G = momepy.gdf_to_nx(network, multigraph=False, directed=True, length='distance')
nx.set_node_attributes(G, {n: n for n in G.nodes}, 'pos')
nx.relabel_nodes(G, {k[1]: v for k, v in nx.get_edge_attributes(G, 'to_node').items()}, False)

# Load pre-computed shortest path travel time between all intersection OD pairs
shortest_times = pd.read_csv(path_time_file, index_col=0)
shortest_times.columns = shortest_times.columns.astype(np.int64)


def plot_depots():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600, tight_layout=True)
    ax.set_aspect('equal')
    nx.draw(G, nx.get_node_attributes(G, 'pos'), arrows=False, node_size=0, edge_color='grey')
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), depots,
                           node_size=5, node_shape='o', node_color='r', label='Depots')
    plt.show()


def plot_stations():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600, tight_layout=True)
    ax.set_aspect('equal')
    nx.draw(G, nx.get_node_attributes(G, 'pos'), arrows=False, node_size=0, edge_color='grey')
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), chargingStations,
                           node_size=5, node_shape='^', node_color='b', label='Stations')
    plt.show()


# TODO: Visualise results on map
# H = graph.copy()
# for n in H.nodes():
#     H.nodes[n]['N'] = 0

# # 1. Add new nodes in H based on Location.loc_from_source and Location.locFromTarget
#     # node = xxx

#     # 2. Overlapped nodes should be marked with a count value
#     if H.has_node(node):
#         node['N'] += 1
#         pass

# # 3. Remove old intersections without any agents
# for n in H.nodes():
#     if H.nodes[n]['N'] == 0:
#         nx.remove_node(n)

# # 4. Visualise overlapped graphs
