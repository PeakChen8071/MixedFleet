import pandas as pd
import numpy as np
import geopandas as gpd
import momepy
import networkx as nx

from Parser import map_file, path_time_file


# Construct Networkx directed graph based on parsed edge list
network = gpd.read_file(map_file)
network = network.to_crs('epsg:2263')  # Manhattan EPSG
G = momepy.gdf_to_nx(network, multigraph=False, directed=True, length='distance')
nx.set_node_attributes(G, {n: n for n in G.nodes}, 'pos')
nx.relabel_nodes(G, {k[1]: v for k, v in nx.get_edge_attributes(G, 'to_node').items()}, False)

# Load pre-computed shortest path travel time between all intersection OD pairs
shortest_times = pd.read_csv(path_time_file, index_col=0)
shortest_times.columns = shortest_times.columns.astype(np.int64)
