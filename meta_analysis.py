"""
Doing some quick tests to learn what meta-paths are acceptable versus not acceptable.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    #do some sort of meta-analysis to limit pathways
    indication_json = "indication_paths.json"
    literature_dir = "./wiki_text"
    if not os.path.exists(indication_json):
        print("Downloading indications file...")
        indication_url = "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.json"
        r = requests.get(indication_url, allow_redirects=True)
        open(indication_json, 'wb').write(r.content)
    

    with open(indication_json, "r") as jfile:
        data = json.load(jfile)
    
    all_connections = []
    for pathway in data:
        connections = []
        links = pathway['links']
        nodes = pathway['nodes']
        for link in links:
            source = link['source']
            target = link['target']

            for node in nodes:
                if node['id'] == source:
                    source_type = node['label']
                elif node['id'] == target:
                    target_type = node['label']
            connections.append([source_type, target_type])
        all_connections.extend(connections)
    connection_strings = []
    for connection in all_connections:
        tmp = "-".join(connection)
        connection_strings.append(tmp)

    unique, unique_counts = np.unique(connection_strings, return_counts=True)
    unique = list(unique)
    unique_counts = list(unique_counts)
    zipped = list(zip(unique_counts, unique))
    zipped.sort(reverse=True)
    unique_counts, unique = zip(*zipped)
    
    for u, uc in zip(unique, unique_counts):
        print(u, uc)
        #sys.exit(0)

    unique = list(unique)

    with open("meta_analysis_pruning.json", "w") as jfile:
        json.dump({"connections": unique}, jfile)

if __name__ == "__main__":
    main()
