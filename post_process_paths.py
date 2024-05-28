"""
Incomplete.
"""
import sys
import networkx as nx
import matplotlib.pyplot as plt

def create_graph():
    G = nx.DiGraph()
    G.add_nodes_from(entities)
    for triple in triplets:
        G.add_edge(triple[0], triple[2], label=triple[1])
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black', font_weight='bold')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    paths = list(nx.all_simple_paths(G, "carbinoxamine", "seasonal allergic conjunctivitis"))
    for path in paths:
        print(path)
    # Save the graph as a PNG file
    plt.savefig("graph.png")
    plt.show()
