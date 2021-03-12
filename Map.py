import networkx as nx
import matplotlib.pyplot as plt

from Parser import read_edgeList, depot_nodes


# Construct Networkx directed graph based on parsed edge list
edgeList, source, target, edge_attr = read_edgeList()
G = nx.convert_matrix.from_pandas_edgelist(edgeList, source, target, edge_attr, nx.DiGraph)
nx.set_node_attributes(G, edgeList['pos'].to_dict(), 'pos')  # Inject node data


def plot_map():
    node_pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600, tight_layout=True)
    ax.set_aspect('equal')
    nx.draw(G, node_pos, arrows=False, node_size=0, edge_color='grey')
    nx.draw_networkx_nodes(G, node_pos, depot_nodes, node_size=5, node_shape='o', node_color='r', label='Depots')
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
