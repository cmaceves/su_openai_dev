import os
import sys
import json
import numpy as np
import pandas as pd
import predict_identfier_embedding

def main():
    directory_name = "/home/caceves/su_openai_dev/no_wiki_text"
    all_json_files = [os.path.join(directory_name, x) for x in os.listdir(directory_name) if 'prompts' not in x]

    not_grounded_nodes = {"node_names":[]}

    for i, json_file in enumerate(all_json_files):
        with open(json_file, 'r') as jfile:
            data = json.load(jfile)

        grounded_nodes = data['grounded_nodes']
        triples = data['triples']

        not_grounded = []
        is_grounded = []
        for key, value in grounded_nodes.items():
            grounded = False
            for k, v in value.items():
                if len([x for x in v if x != '']) != 0:
                    grounded = True
                    is_grounded.append(key)
                    break
            if not grounded:
                not_grounded.append(key)
        print(json_file)
        print("not grounded", not_grounded)
        for ng in not_grounded:
            predict_identfier_embedding.main(ng)
        print(triples)
        sys.exit(0)
        not_grounded_nodes['node_names'].extend(not_grounded)
    return(not_grounded_nodes)

if __name__ == "__main__":
    main()



