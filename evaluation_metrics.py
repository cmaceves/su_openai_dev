"""
Evaluate random indications based on node grodunings and predicates.
"""
import os
import sys
import json
import random

import wiki_pipeline
import post_process_paths
random.seed(42)

def parse_indication_paths(indication_json, n=10):
    """
    indication_json : string
        The full path to the file containing the known indiciation paths.
    n : int 
        The number of indiciations to randomly choose for evaluation.

    Parses the json file containing the indications and randomly selects N for evaluation.
    """
    with open(indication_json, 'r') as jfile:
        data = json.load(jfile)

    selected_items = random.choices(data, k=n)   
    return(selected_items)

def main():
    #develop a system for storing the evaluation including prompts in a dated way

    #model type
    
    #randomly fetch n indications
    n = 10
    indication_json = "indication_paths.json"
    literature_dir = "./wiki_text"
    if not os.path.exists(indication_json):
        print("Downloading indications file...")
        indication_url = "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.json"
        r = requests.get(indication_url, allow_redirects=True)
        open(indication_json, 'wb').write(r.content)
    
    evaluation_indications = parse_indication_paths(indication_json, n)
    
    #leave open for parallel process
    for indication in evaluation_indications:
        node_names = []
        for node in indication['nodes']:
            node_names.append(node['name'].lower())
        print(node_names)
        #wiki_pipeline.main(indication) 
        basename = node_names[0] + "_" + node_names[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)
        print(output_filename)
        with open(output_filename, 'r') as ofile:
            data = json.load(ofile)
        triples = data['triplets']
        entities = data['entities']
        paths = post_process_paths.create_graph(entities, triples, node_names[0], node_names[-1])
        path_lengths = []
        for path in paths:
            if len(path) > 3 and len(path) < 6:
                print(path)
            path_lengths.append(len(path))
        print(len(paths))
        sys.exit(0)
        #evaluate based on node groundings
        #evaluate based on predicate similarity
    

if __name__ == "__main__":
    main()
