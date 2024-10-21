import os
import ast
import sys
import json
import requests
import wikidata.client

from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import prompts
import util
import wiki_pipeline
import numpy as np

name_lookup_url = "https://name-lookup.transltr.io/lookup"
node_norm_url = "https://nodenorm.transltr.io/get_normalized_nodes?curie="
go_types = ["Biological Process", "Cellular Component", "Molecular Activity"]
react_types = ["Pathway"]
mesh_types = ["Chemical Substance", "Disease", "Drug"]
chebi_types = ["Chemical Substance"]
uniprot_types = ["Protein"]
hp_types = ['Phenotypic Feature']

client = wikidata.client.Client()

def find_databases(node_type):
    databases = []
    if node_type in go_types:
        databases.append("GO")
    if node_type in mesh_types:
        databases.append("MESH")
    if node_type in uniprot_types:
        databases.append("UniProt")
    return(databases)

def ground_embedding_space(nodes, mechanism_paragraph, grounded_node_type):
    with open("master_dict_2.json", 'r') as jfile:
        master_dict = json.load(jfile)
    embedding_vectors = np.load('new_openai_embedding.npy')
    
    found_identifiers = {}
    for node in nodes:
        print("node", node)
        node_type = grounded_node_type[node]
        databases = find_databases(node_type)

        resp = prompts.define_gpt(node.lower())
        vec = util.try_openai_embeddings(resp)
        vec = np.array(vec)
        vec = vec.reshape(1, -1)
        cosine_scores = cosine_similarity(vec, embedding_vectors)
        cosine_scores = cosine_scores.flatten()

        indexes = list(range(len(cosine_scores)))
        zipped = list(zip(cosine_scores, indexes))
        zipped.sort(reverse=True)
        cosine_scores, indexes = zip(*zipped)
        cosine_scores = list(cosine_scores)[:20]
        indexes = list(indexes)[:20]
        
        labels = []
        label_string = ""
        counter = 1
        print("databases", databases)
        if len(databases) == 0:
            found_identifiers[node] = ""
            continue

        for j, (key, value) in enumerate(master_dict.items()):
            if j in indexes:
                database_tmp = key.split(":")
            
                if len(database_tmp) == 1:
                    if database_tmp[0].startswith("D"):
                        database_tmp =  "MESH"
                    else:
                        database_tmp = "UniProt"
                else:
                    database_tmp = database_tmp[0]
                if database_tmp not in databases:
                    continue
                
                label_string += str(counter) + "." + value + "\n"
                if database_tmp == "MESH":
                    key = "MESH:" + key
                labels.append(key)
                counter += 1
                if counter > 10:
                    break
                
        resp, prompt = prompts.choose_embedding_identifier(node, label_string)
        try:
            index = int(resp)-1
        except:
            print(resp)
            found_identifiers[node] = ""
            continue
        try:
            found_identifiers[node] = labels[index] 
        except:
            found_identifiers[node] = ""
    return(found_identifiers)

def ground_name_lookup(nodes, mechanism_paragraph, grounded_node_types):
    grounded_nodes = {}
    for node in nodes:
        param  = {"string": node}
        resp = requests.post(name_lookup_url, params=param)
        all_matches = resp.json()
        labels = []
        label_string = ""

        for i,match in enumerate(all_matches):
            label = match['label']
            labels.append(label)
            label_string += str(i+1) + "." + label +"\n"
            print(match)
        sys.exit(0)
        print(node)
        print(label_string)
        response, prompt = prompts.choose_identifier(node, label_string, mechanism_paragraph)
        index = int(response) - 1
        grounded_nodes[node] = all_matches[index]
        print(response)
    sys.exit(0)

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

def wikidata_search_wrapper(term):
    possible_wikidata_matches = {}
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'language': 'en',
        'format': 'json',
        'search': term,
        'limit': 30,
        'type': 'item'
    }

    response = requests.get(url, params=params)
    data = response.json()
    search_results = data['search']
    if len(search_results) == 0:
        return("")
    
    for result in search_results:
        print(term, result)
        label = result['label']
        possible_wikidata_matches[label] = ""
        if 'description' in result:
            possible_wikidata_matches[label] = result['description']
    print("len", len(search_results))
    wikidata_string = ""
    for i, (key, value) in enumerate(possible_wikidata_matches.items()):
        if i != 0:
            wikidata_string += "\n"
        wikidata_string += str(i+1) + ". " + key + " - " + value
    
    resp, prompt = prompts.choose_wikidata_match(term, wikidata_string)
    index = int(resp)-1
    return(search_results[index]['id'])


def wikidata_fetch_wrapper(identifier):
    if identifier == "":
        return({})

    identifier_dictionary = {}
    wiki_entity = client.get(identifier, load=True)
    claims = wiki_entity.data['claims']
    properties = ["P486", "P352", "P5270", "P3841", "P686"]
    for key, value in claims.items():
        if key in properties:
            identifier = value[0]['mainsnak']['datavalue']['value']
            index = properties.index(key)
            if index == 0:
                identifier_dictionary['mesh_id'] = identifier
            elif index == 1:
                identifier_dictionary['uniprot'] = identifier
            elif index == 2:
                identifier_dictionary['mondo'] = identifier
            elif index == 3:
                identifier_dictionary['hpo'] = identifier
            elif index == 4:
                identifier_dictionary['go'] = identifier
    return(identifier_dictionary)

def wikidata_grounding(terms):
    #lemmetaize the scientific phrase
    all_terms = []
    """
    for term in terms:
        resp, prompt = prompts.lemmatize(term)
        
        print(term, resp)
    sys.exit(0)
    """
    #this gets the wikidata id for the best matched page
    best_match_ids = Parallel(n_jobs=5)(delayed(wikidata_search_wrapper)(term) for term in terms) 
    
    #this bit actually searches the page for the identifiers
    structured_identifiers = Parallel(n_jobs=5)(delayed(wikidata_fetch_wrapper)(identifier) for identifier in best_match_ids)

    grounded_dictionary = dict(zip(terms, structured_identifiers))
    return(grounded_dictionary)

def expand_node_terms(nodes, grounded_types, category_messages):
    """
    Given a list of nodes, expand each to contain a list of synonyms.
    """
    expanded_nodes = {}
    for node in nodes:
        type_restriction = grounded_types[node]
        resp, prompt = prompts.alternate_entity_expansion(node, type_restriction, category_messages)
        synonyms = resp.split("\n")
        synonyms = [x.split(". ")[-1] for x in synonyms]
        final_synonyms = []
        for synonym in synonyms:
            resp, prompt = prompts.alternate_equivalency_check(node, synonym)
            if resp == "yes":
                final_synonyms.append(synonym)
        expanded_nodes[node] = final_synonyms
    return(expanded_nodes)

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

def main(indication, predicate_json):
    entity_json = "./predicates/entities.json"
    literature_dir = "./no_wiki_text"
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

    #ground truth values
    node_identifiers = []
    node_names = []
    for node in indication['nodes']:
        node_identifiers.append(node['id'])
        node_names.append(node['name'].lower())
    basename = node_names[0] + "_" + node_names[-1] + ".json"
    basename = basename.replace(" ","_")
    output_filename = os.path.join(literature_dir, basename)
    #if os.path.isfile(output_filename):
    #    return

    all_triples = alternate_mechanism_one_shot(node_names[0], node_names[-1], predicate_data)
    #pull out all nodes for the mechanism
    unique_nodes = []
    for triple in all_triples:
        if triple[0] not in unique_nodes:
            unique_nodes.append(triple[0])
        if triple[-1] not in unique_nodes:
            unique_nodes.append(triple[-1])

    string_version = ""
    for i, triple in enumerate(all_triples):
        if i != 0:
            string_version += ". "
        tmp = " ".join(triple)
        string_version += tmp
    mechanism_paragraph, prompt  = prompts.mech_paragraph_form(string_version)

    category_messages = []
    for k, v in record_category_def.items():
        category_messages.extend(v)
    
    #type ground the nodes 
    grounded_types = ground_node_types(unique_nodes, record_category_def, category_string)
    print(grounded_types)

    #ground_name_lookup(unique_nodes, mechanism_paragraph, grounded_types)
    found_identifiers = ground_embedding_space(unique_nodes, mechanism_paragraph, grounded_types)
    output_data = {"triples":all_triples, "grounded_nodes":found_identifiers, "grounded_types": grounded_types}
    print(found_identifiers)
    with open(output_filename, 'w') as ofile:
        json.dump(output_data, ofile)
    return
    #HERE

    node_dictionary = {}
    for un in unique_nodes:
        node_dictionary[un] = [un]
    
    sys.exit(0)

    first_pass_grounding = wikidata_grounding(unique_nodes)    
    print(first_pass_grounding)
    sys.exit(0)
    ungrounded_nodes = []
    for key, value in first_pass_grounding.items():
        if len(value) == 0:
            ungrounded_nodes.append(key)
            param  = {"string": key}
            resp = requests.post(name_lookup_url, params=param)
            all_matches = resp.json()
            match_string = ""
            for i, match in enumerate(all_matches):
                match_string += str(i+1) + ". " + str(match) + "\n"
            resp, prompt = prompts.alternate_pick_grounding(key, match_string)
            for x in all_matches:
                print(x)

            match  = all_matches[int(resp)-1]['curie']
            print(key, match)            
            sys.exit(0)
    print(all_triples)
    sys.exit(0) 
    return

    first_pass_grounding = ground_node_terms(node_dictionary)
    expand_nodes = []
    for key, value in first_pass_grounding.items():
        if key not in value:
            expand_nodes.append(key)
            continue
        val = value[key]
        if len([x for x in val if x != '']) == 0:
            expand_nodes.append(key)
    
    #expand to synonyms
    expanded_nodes = expand_node_terms(expand_nodes, grounded_types, category_messages)
    grounded_nodes = ground_node_terms(expanded_nodes)
    for key, val in first_pass_grounding.items():
        grounded_nodes[key] = val 

    print(grounded_nodes)    
    output_data = {"triples":all_triples, "expanded_nodes":expanded_nodes, "grounded_nodes":grounded_nodes, "grounded_types": grounded_types}
    sys.exit(0)
    #return
    with open(output_filename, 'w') as ofile:
        json.dump(output_data, ofile)

if __name__ == "__main__":
    main()
