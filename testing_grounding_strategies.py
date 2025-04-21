"""
Use OpenAI, OpenAI Embeddings, and NodeNorm to try and ground human biological processes.
"""
import os
import sys
import ast
import json
import openai
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()
import prompts
import requests
from joblib import Parallel, delayed
from transformers import *
import util
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import torch
import random
random.seed(42)
from evaluation_metrics import search_entity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import color_scheme

figure_dir = "/home/caceves/su_openai_dev/su_figures"
output_dir = "/home/caceves/su_openai_dev/no_wiki_text"
all_json_files = [os.path.join(output_dir, x) for x in os.listdir(output_dir) if x.endswith(".json")]
name_lookup_url = "https://name-lookup.transltr.io/lookup"
node_norm_url = "https://nodenorm.transltr.io/get_normalized_nodes?curie="

indication_json = "indication_paths.json"
evaluation_indications = util.parse_indication_paths(indication_json)
uniprot_json = "/home/caceves/su_openai_dev/gpt_databases_def/uniprot_def_1.json"


def check_label_matching(protein_dict):
    """
    Given an identifier and the DrugMechDB label determine if this label matches the translator label.
    """
    matched_ids = {}
    nonmatched_ids = {}
    for i, (identifier, label) in enumerate(protein_dict.items()):
        if i % 1000 == 0:
            print(i)
        identifier_tmp = identifier.replace("UniProt", "UniProtKB")
        resp = requests.get(node_norm_url+identifier_tmp)
        resp = resp.json()[identifier_tmp]
        eids = [x['label'] for x in resp['equivalent_identifiers'] if 'label' in x]
        eids.append(resp['id']['label'])
        found = False
        for eid in eids:
            if label in eid:
                found=True
                break
        if not found:
            nonmatched_ids[identifier] = label
        else:
            matched_ids[identifier] = label

    print(len(nonmatched_ids))
    print(len(matched_ids))

def find_human_proteins(protein_dict):
    human_proteins = {}
    uniprot_db = "/home/caceves/su_openai_dev/parsed_databases/uniprot_definitions.json"
    with open(uniprot_db, "r") as jfile:
        uniprot_def = json.load(jfile)
    protein_ids = ["UniProt:"+x['accession'] for x in uniprot_def]
    for key, value in protein_dict.items():
        if key in protein_ids:
            human_proteins[key] = value
    return(human_proteins)

def find_drugmechdb_values(indications, label, db):
    node_dict = {}
    for indication in indications:
        for x in indication['nodes']:
            if x['label'] == label:
                if x['id'] not in node_dict and db in x['id']:
                    node_dict[x['id']] = x['name']
    return(node_dict)

def try_scibert(tokenizer, model, target):
    inputs = tokenizer(target, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding_2 = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return(embedding_2)

def return_ground_truth_path(drug, disease, indications):
    for indication in indications:
        if indication['graph']['drug'].lower() == drug.lower() and indication['graph']['disease'].lower() == disease.lower():
            identifiers = [x['id'] for x in indication['nodes']]
            labels = [x['name'] for x in indication['nodes']]
            return(identifiers, labels)
    return(None, None)

def create_batches(token_counts, token_limit=8192):
    batches = []
    current_batch = []
    current_batch_tokens = 0

    for i, token_count in enumerate(token_counts):
        if token_count > token_limit:
            raise ValueError(f"Single input at index {i} exceeds the token limit of {token_limit}.")
        
        # Check if adding this item would exceed the token limit
        if current_batch_tokens + token_count > token_limit:
            # Finalize the current batch
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0
        
        # Add the current item to the batch
        current_batch.append(i)
        current_batch_tokens += token_count

    # Add the last batch if it has any items
    if current_batch:
        batches.append(current_batch)
    
    return batches

def fetch_openai_scores(term, loaded_array, uniprot_order, uniprot_def, n=10):
    """
    """
    #get the top 50 results for close openai matches
    embedding_1 = util.get_embeddings([term])
    embedding_1 = np.array(embedding_1)
    distances = util.calculate_distances(loaded_array, embedding_1)

    zipped = list(zip(distances, uniprot_order, uniprot_def))
    zipped.sort()
    distances, all_keys, uniprot_def = zip(*zipped)
    distances = list(distances)
    all_keys = list(all_keys)
    uniprot_def = list(uniprot_def)

    final_keys = []
    final_defs = []
    for i in range(n):
        final_keys.append(all_keys[i])
        final_defs.append(uniprot_def[i])
    return(final_keys, final_defs)

def fetch_scibert_scores(term, loaded_array, uniprot_order):
    final_keys = []
    embedding_1 = try_scibert(tokenizer, model, term)
    loaded_array = np.squeeze(loaded_array)
    distances = util.calculate_distances(loaded_array, embedding_1)
    #all_keys = list(uniprot_def.keys())
    all_keys = uniprot_order
    zipped = list(zip(distances, all_keys))
    zipped.sort()
    distances, all_keys = zip(*zipped)
    distances = list(distances)
    all_keys = list(all_keys)
    for i in range(10):
        final_keys.append(all_keys[i])
        #final_keys.append(uniprot_order[i])
        #print(distances[i], all_keys[i], uniprot_def[all_keys[i]])
    return(final_keys)

def fetch_translator_scores(node):
    """
    Name lookup followed by node normalization for the name lookup hits.
    """
    final_keys = []
    final_ids = [] 
    param = {"string": node, "only_taxa":"NCBITaxon:9606", "limit":10}
    resp = requests.post(name_lookup_url, params=param)
    all_matches = resp.json()

    for i in range(10):
        if len(all_matches) <= i:
            continue
        final_keys.append(all_matches[i]['label'])
        final_ids.append([all_matches[i]['curie']])
        #print(all_matches[i]['label'], all_matches[i]['curie'])
        #hit node norm to get other identifiers
        resp = requests.get(node_norm_url+all_matches[i]['curie'])
        all_identifiers = resp.json()
        equivalent_ids = all_identifiers[all_matches[i]['curie']]['equivalent_identifiers']
        equivalent_ids = [x['identifier'] for x in equivalent_ids]
        final_ids[i].extend(equivalent_ids)

    return(final_keys, final_ids)

def main():
    """
    Test NodeNorm, OpenAI Chat Completetion, OpenAI Embeddings.
    """
    figure_dir = "/home/caceves/su_openai_dev/su_figures"

    batch_input_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
    embedding_dir = "/home/caceves/su_openai_dev/embeddings"
    parsed_db_dir = "/home/caceves/su_openai_dev/parsed_databases"
    database_name = "go"

    n = 50

    fuzzy_file = "fuzzied_entities.csv"
    create_fuzzy = False    
    if create_fuzzy:
        #fetch biological processes
        drugmechdb_nodes_dict = find_drugmechdb_values(evaluation_indications, "BiologicalProcess", "GO")

        #selected random test set
        random_selection = random.sample(list(drugmechdb_nodes_dict.keys()), n)
        random_entities = {key : drugmechdb_nodes_dict[key] for key in random_selection}

        #fuzzy and dumpy to the file
        random_entity_string = ""
        for i, (key, value) in enumerate(random_entities.items()):
            random_entity_string += str(i+1) + ". " + value + "\n"
        response, prompt = prompts.fuzzy_nodes(random_entity_string)
        response = json.loads(response)
        df = pd.DataFrame({"original": list(response.keys()), "fuzzy":list(response.values()), "ids":list(random_entities.keys())})
        df.to_csv(fuzzy_file)
        print(df)

    #create the biobert numpy array
    create_biobert = False
    if create_biobert:
        #load biobert model
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        scibert_loaded_array = util.load_array("scibert_uniprot_embeddings.npy")
    
    evaluate = False
    if evaluate:
        ground_test_dir = "./test_grounding"
        #fuzzy proteins entities
        df = pd.read_csv(fuzzy_file)

        all_responses = []
        for i, (index, row) in enumerate(df.iterrows()):
            if i % 5 == 0 and i > 0:
                print(i)
            query_entity = row['fuzzy']
            output_dict = {}
            out_filename = os.path.join(ground_test_dir, "%s.json"%(query_entity.replace(" ","_")))
            #check if exists
            if os.path.isfile(out_filename):
                continue

            #openai embedding grounding
            resp, prompt  = prompts.define_nodes(query_entity)
            return_names, return_ids, comparison_scores = evaluation_metrics.search_entity(resp, database_name, 10)
            
            #scibert_final_keys = fetch_scibert_scores(resp, scibert_loaded_array, uniprot_order)

            #translator grounding
            node_norm_keys, node_norm_ids = fetch_translator_scores(query_entity)

            #chat completion techniques
            resp, prompt = prompts.test_node_grounding(database_name, query_entity)


            output_dict['original'] = row['original']
            output_dict['fuzzy'] = row['fuzzy']
            output_dict['ids'] = row['ids']

            output_dict['openai_embedding_ids'] = return_ids
            output_dict['openai_embedding_names'] = return_names

            output_dict['openai_chat_ids'] = ast.literal_eval(resp)

            output_dict['nodenorm_ids'] = node_norm_keys
            output_dict['nodenorm_names'] = node_norm_ids


            with open(out_filename, 'w') as wfile:
                json.dump(output_dict, wfile)

    plot_test = True
    if plot_test:
        ground_test_dir = "./test_grounding"
        #fuzzy proteins entities
        df = pd.read_csv(fuzzy_file)

        #x = ["NameResovler", "OpenAI Chat Completion", "OpenAI Embeddings"]
        x = ["NameResolver", "OpenAI Chat Completion"]
        yes_counts = [0] * len(x)
        no_counts = [0] * len(x)

        yes_counts_first = [0] * len(x)
        no_counts_first = [0] * len(x)
        for i, (index, row) in enumerate(df.iterrows()):
            #if i > 2:
            #    continue
            query_entity = row['fuzzy']
            ground_truth_id = row['ids']
            out_filename = os.path.join(ground_test_dir, "%s.json"%(query_entity.replace(" ","_")))

            #open results file
            with open(out_filename, 'r') as ofile:
                data = json.load(ofile)

            openai_embedding = data['openai_embedding_ids']
            openai_chat = data['openai_chat_ids']
            nodenorm = data['nodenorm_names']

            openai_embedding_names = data['openai_embedding_names']
            nodenorm_names = data['nodenorm_ids']

            """
            print("openai embedding names", openai_embedding_names)
            print("node norm names", nodenorm_names)
            print(query_entity)
            print(row['original'])
            print(ground_truth_id)
            print("openai chat", openai_chat)
            sys.exit(0)
            """
            #top 10 plot
            flattened_nodenorm = [item for sublist in nodenorm for item in sublist]
            #if ground_truth_id in openai_embedding:
            #    yes_counts[2] += 1
            #else:
            #    no_counts[2] += 1
            if ground_truth_id in openai_chat:
                yes_counts[1] += 1
            else:
                no_counts[1] += 1
           
            if ground_truth_id in flattened_nodenorm:
                yes_counts[0] += 1
            else:
                no_counts[0] += 1
    
            continue
            #top 1 plot
            if ground_truth_id == openai_embedding[0]:
                yes_counts_first[2] += 1
            else:
                no_counts_first[2] += 1 
            if ground_truth_id == openai_chat[0]:
                yes_counts_first[1] += 1
            else:
                no_counts_first[1] += 1
            if len(nodenorm) > 0 and ground_truth_id in nodenorm[0]:
                yes_counts_first[0] += 1
            else:
                no_counts_first[0] += 1
    """
    #top 1 plot all methods
    plt.close()
    plt.clf()
    bar_width = 0.8
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, yes_counts_first, label='Correct Label Ranked 1', color=color_scheme.PRIMARY_COLOR, width=bar_width, edgecolor=color_scheme.EDGE_COLOR)
    ax.bar(x, no_counts_first, bottom=yes_counts_first, label='Incorrect Label Ranked 1', color=color_scheme.SECONDARY_COLOR, width=bar_width, edgecolor=color_scheme.EDGE_COLOR)
    ax.set_xticklabels(x, fontsize=14) 
    ax.legend()
    plt.xlabel("Grounding Strategy", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "grounding_test_first_label.png"), dpi=300)
    
    #top 10 plot all methods
    plt.close()
    plt.clf()
    bar_width = 0.8
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, yes_counts, label='Correct Label in Top 10', color=color_scheme.PRIMARY_COLOR, width=bar_width, edgecolor=color_scheme.EDGE_COLOR)
    ax.bar(x, no_counts, bottom=yes_counts, label='Incorrect Label in Top 10', color=color_scheme.SECONDARY_COLOR, width=bar_width, edgecolor=color_scheme.EDGE_COLOR)
    ax.set_xticklabels(x, fontsize=14) 
    ax.legend()
    plt.xlabel("Grounding Strategy", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "grounding_test_larger.png"), dpi=300)
    sys.exit(0)
    """    

    plt.close()
    plt.clf()
    bar_width = 0.8
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, yes_counts, label='Correct Label in Top 10', color=color_scheme.PRIMARY_COLOR, width=bar_width, edgecolor=color_scheme.EDGE_COLOR)
    ax.bar(x, no_counts, bottom=yes_counts, label='Incorrect Label in Top 10', color=color_scheme.SECONDARY_COLOR, width=bar_width, edgecolor=color_scheme.EDGE_COLOR)
    ax.set_xticklabels(x, fontsize=14) 
    ax.legend()
    plt.xlabel("Grounding Strategy", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "grounding_test_smaller.png"), dpi=300)

    print(yes_counts)
    print(no_counts)


if __name__ == "__main__":
    main()
