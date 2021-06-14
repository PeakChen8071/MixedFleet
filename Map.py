import networkx as nx
import matplotlib.pyplot as plt

from Parser import map_file, depot_nodes


# Construct Networkx directed graph based on parsed edge list
G = nx.read_shp(map_file , geom_attrs=False)
nx.set_node_attributes(G, {n: n for n in G.nodes}, 'pos')
nx.relabel_nodes(G, {k[1]: v for k, v in nx.get_edge_attributes(G, 'to_node').items()}, False)


def plot_map():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600, tight_layout=True)
    ax.set_aspect('equal')
    nx.draw(G, nx.get_node_attributes(G, 'pos'), arrows=False, node_size=0, edge_color='grey')
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), depot_nodes,
                           node_size=5, node_shape='o', node_color='r', label='Depots')
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
