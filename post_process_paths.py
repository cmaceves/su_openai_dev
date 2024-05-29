"""
Construct the graphs and sort through the paths to find the most likely solution.
"""
import os
import sys
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import wiki_pipeline

def create_graph(entities, triplets, sub, obj):
    G = nx.DiGraph()
    G.add_nodes_from(entities)
    for triple in triplets:
        G.add_edge(triple[0], triple[2], label=triple[1])

    """
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black', font_weight='bold')
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # Save the graph as a PNG file
    plt.savefig("graph.png")
    plt.show()
    """ 
    paths = list(nx.all_simple_paths(G, "carbinoxamine", "seasonal allergic conjunctivitis"))
    return(paths)

def main():
    indication_json = "./indication_paths.json"
    solution_filename = "data.tsv"
    literature_dir = "./wiki_text"
    solution_df = pd.read_table(solution_filename) 
    solution_names = wiki_pipeline.parse_solutions(indication_json, solution_df)
    for i, (index, row) in enumerate(solution_df.iterrows()):
        print(i, "of", len(solution_df), "done.")
        all_triples = []
        all_entities = []
        all_prompts = []
        solution_steps = solution_df.iloc[index].tolist()
        solution_identifiers = [x.lower() for x in solution_names[i]]
        if len(solution_identifiers) == 0:
            continue
        basename = solution_identifiers[0] + "_" + solution_identifiers[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)
        if not os.path.isfile(output_filename):
            continue
        with open(output_filename, 'r') as jfile:
            data = json.load(jfile)
        print(data.keys())
        sys.exit(0) 
    
        all_paths = create_graphs(entities, triplets, sub, obj)


if __name__ == "__main__":
    main()
