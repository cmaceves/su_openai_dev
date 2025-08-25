import os
import ast
import sys
import json
import requests

from itertools import combinations
import prompts
import util
import numpy as np
import matplotlib.pyplot as plt

node_types = ['Biological Process', 'Cell', 'Cellular Component', 'Chemical Substance', 'Disease', 'Drug', 'Gene Family', "Gross Anatomical Structure", "Macromolecular Structure", "Molecular Activity", "Organism Taxon", "Pathway", "Phenotypic Feature", "Protein"]

def generate_pairs(terms):
    # Generate all pairwise combinations
    pairs = [" ".join(pair) for pair in combinations(terms, 2)]
    return pairs

def main2(drug, disease, indication, literature_dir, i, predicate_json):
    category_definitions = "./predicates/category_definitions.json"
    name_lookup_url = "https://name-lookup.transltr.io/lookup"

    with open(predicate_json, "r") as jfile:
        predicate_data = json.load(jfile)

    if not os.path.exists(literature_dir):
        os.makedirs(literature_dir)
        print(f"Directory '{literature_dir}' was created.")
    else:
        print(f"Directory '{literature_dir}' already exists.")

    output_filename = os.path.join(literature_dir, "indication_%s.json" %str(i))
    #if os.path.isfile(output_filename):
    #    return

    predicate_string = ""
    for i, k in enumerate(predicate_data.keys()):
        predicate_string += str(i+1) + ". " + k + "\n"

    output_data = {}
    resp, prompt = prompts.one_shot(drug, disease, "gpt-5-nano", predicate_string, node_types)
    try:
        output_data = ast.literal_eval(resp)
    except:
        return
    output_data['grounding'] = {}
    output_data["indication"] = indication
    output_data['drug'] = drug
    output_data['disease'] = disease

    index_map = util.load_index_map(index_path)
    reverse_map = []
    for file, count in index_map:
        reverse_map.extend([(file, i) for i in range(count)])
    total_rows = sum(count for _, count in index_map)
    first_file = index_map[0][0]
    arr = np.load(first_file, mmap_mode='r')
    embedding_dim = arr.shape[1]

    for i, entity in enumerate(output_data['entities']):
        resp, prompt = prompts.define_entity(entity, 'gpt-5-nano')
        output_data['description'] = resp

        return_names, return_ids, comparison_scores, top_embedding = util.search_entity(resp, 10, merged_path='', index_path='', real_vectors=None, reverse_map=reverse_map, embedding_dim=embedding_dim, total_rows=total_rows)

        output_data['grounding'][entity] = {}
        output_data['grounding'][entity]['names'] = return_names
        output_data['grounding'][entity]['ids'] = return_ids
        output_data['grounding'][entity]['scores'] = comparison_scores
        output_data['grounding'][entity]['embedding'] = top_embedding

    with open(output_filename, 'w') as ofile:
        json.dump(output_data, ofile)
    sys.exit(0)
