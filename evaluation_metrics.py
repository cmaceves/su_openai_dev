"""
Evaluate random indications based on node groundings
"""
import os
import ast
import sys
import json
import random
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util
import prompts
import wiki_pipeline
import no_wiki_pipeline
from line_profiler import LineProfiler

random.seed(42)

def main():
    name_lookup_url = "https://name-lookup.transltr.io/lookup"
    predicate_definition_file = "./predicates/predicate_definitions.json" #created by ChatGPT

    #randomly fetch n indications
    n = 50
    indication_json = "indication_paths.json"
    literature_dir = "./no_wiki_text"

    with open(predicate_definition_file, 'r') as pfile:
        predicate_definitions = json.load(pfile)

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

    evaluation_indications = util.parse_indication_paths(indication_json, n)
    

    for i, indication in enumerate(evaluation_indications):
        node_names = []
        node_ids = []
        for node in indication['nodes']:
            node_names.append(node['name'].lower())
        basename = node_names[0] + "_" + node_names[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)
        no_wiki_pipeline.main(indication, predicate_definition_file)

 
if __name__ == "__main__":
    main()
