import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from Configuration import configs


def build_map():
    edge_list_df = pd.read_csv(configs['map_file'])
    edge_list_df.set_index('to_node', drop=False, inplace=True)
    edge_list_df['pos'] = tuple(zip(edge_list_df['to_node_lon'], edge_list_df['to_node_lat']))

    # Construct Networkx directed graph based on edge list
    graph = nx.convert_matrix.from_pandas_edgelist(edge_list_df, 'from_node', 'to_node',
                                                   edge_attr=['length', 'travel_time'],
                                                   create_using=nx.DiGraph)
    nx.set_node_attributes(graph, edge_list_df['pos'].to_dict(), 'pos')  # Inject node data
    return graph


def plot_map(graph):
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
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600, tight_layout=True)
    ax.set_aspect('equal')
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), arrows=False, node_size=0, edge_color='grey')
    # nx.draw_networkx_nodes(H, pos=nx.get_node_attributes(H,'pos'), node_size=2, node_shape='o', node_color='r')
    plt.show()


G = build_map()
