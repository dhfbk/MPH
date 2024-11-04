import networkx as nx
from networkx import Graph
import itertools

def extract_graphs(data):
    spk_agents = []
    for i in range(len(data)):
        spk_agents.append(data[i].spk_agents)

    adr_agents = []
    for i in range(len(data)):
        adr_agents.append(data[i].adr_agents)

    data_adj_list = []

    for orig_list, dest_list in zip(spk_agents, adr_agents):
        last_diff_user = 16
        last_user = 16
        adj_list = []
        for orig, dest in zip(orig_list, dest_list):
            if last_user != orig:
                last_diff_user = last_user
            if dest == 17:
                dest = last_diff_user
            adj_list.append([orig, dest])
            last_user = orig
        data_adj_list.append(adj_list)

    clean_data_adj_list = []

    for adj in data_adj_list:
        clean_adj = []
        for d in  adj:
            if d[1] != -1 and d[1] != d[0]:
                clean_adj.append(d)
        clean_data_adj_list.append(clean_adj)

    graph_set = []
    for adj in clean_data_adj_list:
        G = nx.DiGraph()
        G.add_edges_from(adj)
        if G.has_node(16):
            G.remove_node(16)

        if G.has_node(17):
            G.remove_node(17)
        graph_set.append(G)

    #for each graph in the graph_set, if not present, add node from id 0 to 3
    for G in graph_set:
        for i in range(4):
            if i not in G.nodes:
                G.add_node(i)

    indeces_connected_graphs = []
    for i, G in enumerate(graph_set):
        G = G.to_undirected()
        if nx.is_connected(G):
            indeces_connected_graphs.append(i)

    #given a graph G and a graphlet g, find all instances of g in G (with same number of nodes)
    return graph_set, indeces_connected_graphs

def extract_graphs(data):
    spk_agents = []
    for i in range(len(data)):
        spk_agents.append(data[i].spk_agents)

    adr_agents = []
    for i in range(len(data)):
        adr_agents.append(data[i].adr_agents)

    data_adj_list = []

    for orig_list, dest_list in zip(spk_agents, adr_agents):
        last_diff_user = 16
        last_user = 16
        adj_list = []
        for orig, dest in zip(orig_list, dest_list):
            if last_user != orig:
                last_diff_user = last_user
            if dest == 17:
                dest = last_diff_user
            adj_list.append([orig, dest])
            last_user = orig
        data_adj_list.append(adj_list)

    clean_data_adj_list = []

    for adj in data_adj_list:
        clean_adj = []
        for d in  adj:
            if d[1] != -1 and d[1] != d[0]:
                clean_adj.append(d)
        clean_data_adj_list.append(clean_adj)

    graph_set = []
    for adj in clean_data_adj_list:
        G = nx.DiGraph()
        G.add_edges_from(adj)
        if G.has_node(16):
            G.remove_node(16)

        if G.has_node(17):
            G.remove_node(17)
        graph_set.append(G)

    #for each graph in the graph_set, if not present, add node from id 0 to 3
    for G in graph_set:
        for i in range(4):
            if i not in G.nodes:
                G.add_node(i)

    indeces_connected_graphs = []
    for i, G in enumerate(graph_set):
        G = G.to_undirected()
        if nx.is_connected(G):
            indeces_connected_graphs.append(i)

    #given a graph G and a graphlet g, find all instances of g in G (with same number of nodes)
    return graph_set, indeces_connected_graphs

def extract_graphs_with_prev(data):
    spk_agents = []
    for i in range(len(data)):
        spk_agents.append(data[i].spk_agents)

    adr_agents = []
    for i in range(len(data)):
        adr_agents.append(data[i].adr_agents)

    data_adj_list = []

    for orig_list, dest_list in zip(spk_agents, adr_agents):
        last_diff_user = 17
        last_user = 17
        adj_list = []
        for orig, dest in zip(orig_list, dest_list):
            if last_user != orig:
                last_diff_user = last_user
            if dest == 17:
                dest = last_diff_user
            adj_list.append([orig, dest])
            last_user = orig
        data_adj_list.append(adj_list)

    clean_data_adj_list = []

    for adj in data_adj_list:
        clean_adj = []
        for d in  adj:
            if d[1] != -1 and d[1] != d[0]:
                clean_adj.append(d)
        clean_data_adj_list.append(clean_adj)

    graph_set = []

    for adj in clean_data_adj_list:
        G = nx.DiGraph()
        G.add_edges_from(adj)
        graph_set.append(G)
    #for each graph in the graph_set, if not present, add node from id 0 to 3
    for G in graph_set:
        for i in range(4):
            if i not in G.nodes:
                G.add_node(i)

    indeces_connected_graphs = []
    for i, G in enumerate(graph_set):
        G = G.to_undirected()
        if nx.is_connected(G):
            indeces_connected_graphs.append(i)

    #given a graph G and a graphlet g, find all instances of g in G (with same number of nodes)
    return graph_set, indeces_connected_graphs



def find_undirected_graphlet_instances(G, g):
    instances = 0
    #extract all possible subgraphs of same number of nodes as g.nodes in G
    node_combination = []

    combination = [] # empty list
    for r in range(3, 5):
        # to generate combination
        combination.extend(itertools.combinations(G.nodes, r))

    subgraphs = [nx.induced_subgraph(G, comb) for comb in combination]

    for subgraph in subgraphs:
        if nx.is_isomorphic(Graph(subgraph), g):
            instances = instances + 1

    return instances

#given a graph G and a list of graphlets, find all instances of each graphlet in G
def find_all_undirected_graphlet_instances(G, graphlets):
    instances = []
    number_of_nodes = []
    for graphlet in graphlets:
        instances.append(find_undirected_graphlet_instances(G.to_undirected(), graphlet))
        number_of_nodes.append(len(graphlet.nodes))
    return instances, number_of_nodes

def find_orbits_4(G, graphlets):
    instances = []
    number_of_nodes = []
    for graphlet in graphlets:
        instances.append(find_undirected_graphlet_instances(G.to_undirected(), graphlet))
        number_of_nodes.append(len(graphlet.nodes))
    return instances, number_of_nodes

def find_directed_graphlet_instances(G, g):
    instances = 0
    #extract all possible subgraphs of same number of nodes as g.nodes in G
    combination = [] # empty list
    for r in range(2, 4):
        # to generate combination
        combination.extend(itertools.combinations(G.nodes, r))

    subgraphs = [nx.induced_subgraph(G, comb) for comb in combination]

    for subgraph in subgraphs:
        if nx.is_isomorphic(nx.DiGraph(subgraph), g):
            instances = instances + 1

    return instances

#given a graph G and a list of graphlets, find all instances of each graphlet in G
def find_all_directed_graphlet_instances(G, graphlets):
    instances = []
    number_of_nodes = []
    for graphlet in graphlets:
        instances.append(find_directed_graphlet_instances(G, graphlet))
        number_of_nodes.append(len(graphlet.nodes))
    return instances, number_of_nodes

#given all instances of each graphlet in G, return a list of all instace divided by the number of all possible instances with the same number of nodes

def find_possible_n_nodes(G, n):
    acc = 1
    div = 1
    for i in range(n):
        acc = acc * (len(G.nodes) - i)
        div = div * (i + 1)

    return acc / div

def find_possible_n_nodes2(tot, n):
    acc = 1
    div = 1
    for i in range(n):
        acc = acc * (tot - i)
        div = div * (i + 1)

    return acc / div


def get_instance_ratios(instances, G, number_of_nodes):
    ratios = []
    for instance, n in zip(instances, number_of_nodes):
        ratios.append(instance / find_possible_n_nodes(G, n))
    return ratios