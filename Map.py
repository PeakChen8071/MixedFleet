import pandas as pd
import networkx as nx
import utm
import matplotlib.pyplot as plt

from Configuration import configs


def build_map():
    edge_list_df = pd.read_csv(configs['map_file'])
    graph = nx.convert_matrix.from_pandas_edgelist(edge_list_df, 'from_node', 'to_node',
                                                   edge_attr=['length', 'travel_time'],
                                                   create_using=nx.DiGraph)
    headers = edge_list_df.columns.to_list()
    for n in graph.nodes():
        if n in edge_list_df['from_node'].values:
            graph.nodes[n]['pos'] = tuple(edge_list_df[edge_list_df['from_node'] == n].iloc[0, [
                headers.index('from_node_lon'), headers.index('from_node_lat')]])
        elif n in edge_list_df['to_node'].values:
            graph.nodes[n]['pos'] = tuple(edge_list_df[edge_list_df['to_node'] == n].iloc[0, [
                headers.index('to_node_lon'), headers.index('to_node_lat')]])
        graph.nodes[n]['xy'] = utm.from_latlon(graph.nodes[n]['pos'][1], graph.nodes[n]['pos'][0])[:2]
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
