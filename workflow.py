import os
import ast
import sys
import json
import requests

import itertools
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import prompts
import util
import wiki_pipeline
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

name_lookup_url = "https://name-lookup.transltr.io/lookup"
node_norm_url = "https://nodenorm.transltr.io/get_normalized_nodes?curie="
go_types = ["Biological Process", "Cellular Component", "Molecular Activity"]
react_types = ["Pathway"]
mesh_types = ["Chemical Substance", "Disease", "Drug"]
chebi_types = ["Chemical Substance"]
uniprot_types = ["Protein"]
hp_types = ['Phenotypic Feature']
node_types = ['Biological Process', 'Cell', 'Cellular Component', 'Chemical Substance', 'Disease', 'Drug', 'Gene Family', "Gross Anatomical Structure", "Macromolecular Structure", "Molecular Activity", "Organism Taxon", "Pathway", "Phenotypic Feature", "Protein"]


def extract_pathway_pairs(pathways):
    all_pairs = []
    for pathway in pathways:
        pairs = [(pathway[i], pathway[i + 1]) for i in range(len(pathway) - 1)]
        for pair in pairs:
            if pair in all_pairs:
                continue
            else:
                all_pairs.append(pair)

    all_pairs_strings = []
    for pair in all_pairs:
        tmp = pair[0] + " " + pair[1]
        all_pairs_strings.append(tmp)

    return(all_pairs_strings, all_pairs)

def generate_pairs(terms):
    # Generate all pairwise combinations
    pairs = [" ".join(pair) for pair in combinations(terms, 2)]
    return pairs

def ground_predicates(triples, predicates):
    predicate_string = ""
    messages = []
    predicate_values = []
    for i, (pred, value) in enumerate(predicates.items()):
        predicate_string += str(i+1) + '. ' + pred + '\n'
        messages.extend(value)
        predicate_values.append(pred)

    print(predicate_string)
    for triple in triples:
        resp, prompt = prompts.alternate_ground_predicates(triple[0], triple[-1], triple[1], [], predicate_string)
        print(triple)
        print(resp)
        sys.exit(0)

def ground_node_types(unique_nodes, record_category_def, category_string):
    grounded_types = {}
    category_messages = []
    values = []
    for k, v in record_category_def.items():
        category_messages.extend(v)
        values.append(k)
    for node in unique_nodes:
        resp, prompt = prompts.alternate_categorize_node(node, category_messages, category_string)
        index = int(resp)
        grounded_types[node] = values[index]
    return(grounded_types)

def ground_node_terms(expanded_nodes):
    """
    Ground the node terms against known databases.
    """
    grounded_nodes = {}
    synonym_pages_dict = {}
    pooled_synonyms = []
    
    for entity, synonym_list in expanded_nodes.items():
        pooled_synonyms.extend(synonym_list)

    synonym_pages = Parallel(n_jobs=10)(delayed(wiki_pipeline.matched_wikipedia_pages)(synonym) for synonym in pooled_synonyms)
    tmp_dict = {}
    for synonym_search in synonym_pages:
        if len(synonym_search['pages']) > 0:
            tmp_dict[synonym_search['entity']] = synonym_search['pages']
    for entity, synonym_list in expanded_nodes.items():
        synonym_pages_dict[entity] = []
        for key, value in tmp_dict.items():
            if key in synonym_list:
                if entity not in synonym_pages_dict:
                    synonym_pages_dict[entity] = []
                synonym_pages_dict[entity].extend(value)
    synonym_groundings = {}
    for key, value in synonym_pages_dict.items():
        if key not in synonym_groundings:
            synonym_groundings[key] = {}
        for entity in value:
            if entity in synonym_groundings[key]:
                continue
            mesh_id, uniprot, mondo, hpo, go = wiki_pipeline.ground_synonym(entity)
            synonym_groundings[key][entity] = [mesh_id, uniprot, mondo, hpo, go]
    """
    for key, value in synonym_groundings.items():
        if len(value) == 0:
            param  = {"string": key}
            resp = requests.post(name_lookup_url, params=param)
            all_matches = resp.json()
            label_string = ""
            for i, match in enumerate(all_matches):
                label_string += str(i+1) + "." + match['label'].lower() + "\n"
            resp, prompt = prompts.additional_grounding(label_string, key)
            try:
                index_resp = int(resp) - 1
            except:
                print("fail here")
                continue
            grounded_label = all_matches[index_resp]['curie']
            synonym_groundings[key][key] = [grounded_label]
    """
    for key, value in synonym_groundings.items():
        if len(value) == 0:
            pass
    return(synonym_groundings)

def alternate_mechanism_one_shot(term1, term2, predicate_data):
    """
    Get the mechanism without providing ChatGPT with additional data.
    """
    predicate_string = ""
    for key, value in predicate_data.items():
        predicate_string += key + "\n"
        
    all_triples = []
    resp, prompt = prompts.alternate_mechanism_one_shot(term1, term2, predicate_string)
    print(resp)
    steps = resp.split("\n")
    steps = [x.split(". ")[-1] for x in steps]
    #expand the steps to triples
    for step in steps:
        triple = step.split(" -> ")
        all_triples.append(triple)
    return(all_triples)

def main(drug, disease, indication, predicate_json):
    entity_json = "./predicates/entities.json"
    literature_dir = "./no_follow"
    category_definitions = "./predicates/category_definitions.json"
    name_lookup_url = "https://name-lookup.transltr.io/lookup"

    with open(category_definitions, "r") as cfile:
        record_category_def = json.load(cfile)
    with open(entity_json, "r") as jfile:
        entity_data = json.load(jfile)
    with open(predicate_json, "r") as jfile:
        predicate_data = json.load(jfile)

    if not os.path.exists(literature_dir):
        os.makedirs(literature_dir)
        print(f"Directory '{literature_dir}' was created.")
    else:
        print(f"Directory '{literature_dir}' already exists.")

    category_string = ""
    categories = []
    
    for i, t in enumerate(record_category_def.keys()):
        category_string += str(i) + "." + t + "\n"
        categories.append(t)

    basename = drug + "_" + disease + ".json"
    basename = basename.replace(" ","_")
    output_filename = os.path.join(literature_dir, basename)
    #given the new list of entities, generate a mechanism of action paragraph description
    mechanism_paragraph, resp = prompts.chain_of_thought_4(drug, disease)

    print(mechanism_paragraph)

    predicate_string = ""
    for i, k in enumerate(predicate_data.keys()):
        predicate_string += str(i+1) + ". " + k + "\n"
    
    mp = mechanism_paragraph
    entity_string, prompt = prompts.extract_entities(mp)
    entities = entity_string.split("\n")
    entities = [x.split(". ")[-1].lower() for x in entities if x != ""]
    if drug.lower() not in entities:
        entities.append(drug.lower())
    if disease.lower() not in entities:
        entities.append(disease.lower())

    entity_string = ""
    for i, entity in enumerate(entities):
        entity_string += str(i+1) + ". " + entity + "\n"

    entity_definitions = {}
    entity_types = {}
    print("\n")
    print(entity_string)
    for entity in entities:
        definition, prompt = prompts.define_nodes(entity)
        index, prompt = prompts.type_nodes(entity + "-" + definition)
        try:
            index = int(index)-1
            node_type = node_types[index] 
        except:
            continue
        entity_types[entity] = node_type
        entity_definitions[entity] = definition

    resp, prompt = prompts.chain_of_thought_5(mp, predicate_string, disease, entity_string)
    triples = []

    tmp_triples = resp.split("\n")
    tmp_triples = [x.split(". ")[-1].lower() for x in tmp_triples]
    triples.extend(tmp_triples)
    for i, target in enumerate(entities):
        resp, prompt = prompts.chain_of_thought_5(mp, predicate_string, target, entity_string)
        tmp_triples = resp.split("\n")
        tmp_triples = [x.split(". ")[-1].lower() for x in tmp_triples]
        triples.extend(tmp_triples)

    triple_string = ""
    for i, trip in enumerate(triples):
        triple_string += str(i+1) + ". " + trip.lower() + "\n"

    graph = prompts.get_pathway(drug, disease, mp, triple_string)
    output_data = {"triples": triples, "entities": entities, "mechanism_paragraph": mp, "graph": graph, "entity_types":entity_types, "entity_definitions":entity_definitions, "drug": drug, "disease":disease, "indication":indication}

    with open(output_filename, 'w') as ofile:
        json.dump(output_data, ofile)

if __name__ == "__main__":
    main()
