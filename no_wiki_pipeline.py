import os
import ast
import sys
import json
import requests

from joblib import Parallel, delayed

import prompts
import wiki_pipeline
name_lookup_url = "https://name-lookup.transltr.io/lookup"

def ground_go_terms(entity):
    pass

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
        #in the event that we can't ground using wikipedia, we use name lookup
        for entity in value:
            if entity in synonym_groundings[key]:
                continue
            mesh_id, uniprot, mondo, hpo = wiki_pipeline.ground_synonym(entity)
            synonym_groundings[key][entity] = [mesh_id, uniprot, mondo, hpo]

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
    return(synonym_groundings)
    
def expand_node_terms(nodes):
    """
    Given a list of nodes, expand each to contain a list of synonyms.
    """
    expanded_nodes = {}
    for node in nodes:
        resp, prompt = prompts.alternate_entity_expansion(node)
        synonyms = resp.split("\n")
        synonyms = [x.split(". ")[-1] for x in synonyms]
        synonyms.append(node)
        expanded_nodes[node] = synonyms
    return(expanded_nodes)

def alternate_mechanism(term1, term2):
    """
    Get the mechanism without providing ChatGPT with additional data.
    """
    print(term1, term2)
    all_triples = []
    resp, prompt = prompts.alternate_mechanism(term1, term2)
    steps = resp.split("\n")
    steps = [x.split(". ")[-1] for x in steps]

    #expand the steps to triples
    for step in steps:
        triple = step.split(" -> ")
        all_triples.append(triple)
    return(all_triples)

def main(indication):
    entity_json = "./predicates/entities.json"
    predicate_json = "./predicates/predicates.json"
    literature_dir = "./wiki_text"

    with open(entity_json, "r") as jfile:
        entity_data = json.load(jfile)
    with open(predicate_json, "r") as jfile:
        predicate_data = json.load(jfile)

    if not os.path.exists(literature_dir):
        os.makedirs(literature_dir)
        print(f"Directory '{literature_dir}' was created.")
    else:
        print(f"Directory '{literature_dir}' already exists.")

    #ground truth values
    node_identifiers = []
    node_names = []
    for node in indication['nodes']:
        node_identifiers.append(node['id'])
        node_names.append(node['name'].lower())
    basename = node_names[0] + "_" + node_names[-1] + ".json"
    basename = basename.replace(" ","_")
    output_filename = os.path.join(literature_dir, basename)
    if os.path.isfile(output_filename):
        return
    all_triples = alternate_mechanism(node_names[0], node_names[-1])

    #pull out all nodes for the mechanism
    unique_nodes = []
    for triple in all_triples:
        if triple[0] not in unique_nodes:
            unique_nodes.append(triple[0])
        if triple[-1] not in unique_nodes:
            unique_nodes.append(triple[-1])
    
    expanded_nodes = expand_node_terms(unique_nodes)
    grounded_nodes = ground_node_terms(expanded_nodes)

    output_data = {"triples":all_triples, "expanded_nodes":expanded_nodes, "grounded_nodes":grounded_nodes}
    with open(output_filename, 'w') as ofile:
        json.dump(output_data, ofile)

if __name__ == "__main__":
    main()
