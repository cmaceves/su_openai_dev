"""
Evaluate random indications based on node grodunings and predicates.
"""
import os
import sys
import json
import random
import numpy as np
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
    
    #load the meta-analysis of drug mech db
    metaanalysis_file = "meta_analysis_pruning.json"
    with open(metaanalysis_file, "r") as mfile:
        data = json.load(mfile)['connections']
    acceptable_directed_pairs = [x.split("-") for x in data]
    tmp = []
    for pair in acceptable_directed_pairs:
        pair_1 = wiki_pipeline.camel_to_words(pair[0])
        pair_2 = wiki_pipeline.camel_to_words(pair[1])
        tmp.append([pair_1, pair_2])
    acceptable_directed_pairs = tmp
    flatten_meta_categories = []
    for pair in acceptable_directed_pairs:
        flatten_meta_categories.extend(pair)
    flatten_meta_categories = list(np.unique(flatten_meta_categories))
    flatten_meta_categories = [wiki_pipeline.camel_to_words(x) for x in flatten_meta_categories]

    evaluation_indications = parse_indication_paths(indication_json, n)
    
    #leave open for parallel process
    for indication in evaluation_indications:
        node_names = []
        for node in indication['nodes']:
            node_names.append(node['name'].lower())
        print(node_names)
        
        #actually does the work to pull out the graph
        #wiki_pipeline.main(indication, flatten_meta_categories) 
        #sys.exit(0)

        basename = node_names[0] + "_" + node_names[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)

        with open(output_filename, 'r') as ofile:
            data = json.load(ofile)
        triples = data['triplets']
        entities = data['entities']
        grounded_type = data['grounded_type']
        paths = post_process_paths.create_graph(entities, triples, node_names[0], node_names[-1])
        slimmed_paths = []
        path_lengths = []

        #print(triples)
        print(len(paths))
        for path in paths:
            path_types = []
            for node in path:                
                types = list(np.unique(grounded_type[node]))
                if len(types) != 1:
                    continue
                path_types.append(types[0])
            if len(path_types) != len(path):
                continue
            path_types = [x.replace("biolink:","") for x in path_types]
            expand_pairs = []
            for i, path_val in enumerate(path_types):
                if i+1 < len(path_types):
                    expand_pairs.append([path_val, path_types[i+1]])
            keep = True
            for pair in expand_pairs:
                if pair not in acceptable_directed_pairs:
                    keep = False
                    break
            if keep:
                slimmed_paths.append(path)

        print(len(slimmed_paths))
        print(slimmed_paths[:10])
        print(triples)
        string_paths = ""
        for j, path in enumerate(slimmed_paths):
            construct_string = ""
            node_pairs = []
            for k, node in enumerate(path):
                if k+1 < len(path):
                    node_pairs.append([node, path[k+1]])
            
            for k, node in enumerate(node_pairs):
                construct_string += node[0]
                for trip in triples:
                    if node[0] == trip[0] and node[-1] == trip[-1]:
                        construct_string += " " + trip[1] + " " + trip[-1] + ", " 
                        break
            #construct_string += path[-1]
            string_paths += str(j+1) + "." + construct_string + "\n"
        print(string_paths)
        prompts.choose_path(string_paths, mechanism)
        sys.exit(0)
        #evaluate based on node groundings
        #evaluate based on predicate similarity
    

if __name__ == "__main__":
    main()
