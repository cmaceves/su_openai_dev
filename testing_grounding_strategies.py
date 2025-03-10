"""
Use OpenAI, BioBERT, and NodeNorm to try and ground human proteins.
"""
import os
import sys
import json
import openai
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()
import time
import prompts
import requests
import tiktoken
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

#load the scibert information
#tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
#model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

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

def find_proteins(indications):
    protein_dict = {}
    for indication in indications:
        for x in indication['nodes']:
            if x['label'] == "Protein":
                if x['id'] not in protein_dict:
                    protein_dict[x['id']] = x['name']
    return(protein_dict)

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

def calculate_tokens_per_chunk(texts, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    token_counts = [len(encoding.encode(text)) for text in texts]
    return token_counts

def save_array(array, filename):
    np.save(filename, array)

def load_array(filename):
    return np.load(filename)

# Calculate distances between target vector and all vectors in the array
def calculate_distances(array, target_vector, metric='euclidean'):
    distances = cdist(array, target_vector, metric=metric)
    return distances.flatten()

def get_embeddings(texts, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=texts, model=model)
    embeddings = [res.embedding for res in response.data]
    return embeddings

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
    embedding_1 = get_embeddings([term])
    embedding_1 = np.array(embedding_1)
    distances = calculate_distances(loaded_array, embedding_1)

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
    distances = calculate_distances(loaded_array, embedding_1)
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
    """
    try:
        for i,item in enumerate(all_matches):
            all_labels += str(i+1) + ". " + item['label'] + "\n"
        best_match, prompt  = prompts.determine_translator_synonym(node, all_labels)
        best_match = int(best_match)-1
    except:
        print(drug, disease)
        continue
    answer = all_matches[best_match]
    initial_id = answer['curie']

    #hit node norm to get other identifiers
    resp = requests.get(node_norm_url+initial_id)
    all_identifiers = resp.json()
    equivalent_ids = all_identifiers[initial_id]['equivalent_identifiers']
    equivalent_ids = [x['identifier'] for x in equivalent_ids]
    node_identifiers[node] = equivalent_ids
    """

def main():
    figure_dir = "/home/caceves/su_openai_dev/su_figures"
    protein_dict = find_proteins(evaluation_indications)
    human_proteins = find_human_proteins(protein_dict)
    #none of the main labels perfectly match
    #check_label_matching(human_proteins)
    #sys.exit(0) 

    #selected random test set
    random_selection = random.sample(list(human_proteins.keys()), 50)
    random_proteins = {key : human_proteins[key] for key in random_selection}

    openai_loaded_array = load_array("openai_uniprot_embeddings.npy")
    scibert_loaded_array = load_array("scibert_uniprot_embeddings.npy")

    #uniprot names
    database_name = "openai_uniprot_records.json"
    with open(database_name, "r") as dfile:
        data = json.load(dfile)
    uniprot_order = [] 
    uniprot_def = []
    for item in data:
        uniprot_order.append(item['accession'])
        uniprot_def.append(item['name'] + ":" + item['function'])

    #fuzzy proteins entities
    df = pd.read_csv("protein_matching.csv")
    entities = df['fuzzy'].tolist()
    uniprot_ids = df['uniprot_id'].tolist()

    node_norm_found = 0
    biobert_found = 0
    openai_found = 0
    correct_picked = 0
 
    yes_results = [0, 0, 0]
    no_results = [0, 0, 0]

    yes_results_first = [0, 0, 0]
    no_results_first = [0, 0, 0]

    #this is for selecting the right answer from a small pool
    yes_results_part_2 = [0, 0]
    no_results_part_2 = [0, 0]

    for i, (uniprot_id, entity) in enumerate(zip(uniprot_ids, entities)):
        if i % 5 == 0:
            print(i)
        tmp_id = uniprot_id.replace("UniProt:", "")        
        new_filename = os.path.join("./grounding_results", entity.lower().replace(" ","_")+".json")
        if os.path.isfile(new_filename):
            continue
   
        output_dict = {}
        resp = prompts.define_gpt(entity)
        openai_final_keys, openai_final_defs = fetch_openai_scores(resp, openai_loaded_array, uniprot_order, uniprot_def, entity)
        
        scibert_final_keys = fetch_scibert_scores(resp, scibert_loaded_array, uniprot_order)
        node_norm_keys, node_norm_ids = fetch_translator_scores(entity)
         
        if len(node_norm_ids) > 0:
            node_norm_first = [x.replace("UniProtKB:","") for x in node_norm_ids[0]]
            node_norm_matches = [item for sublist in node_norm_ids for item in sublist]
            node_norm_matches = [x.replace("UniProtKB:","") for x in node_norm_matches]
        else:
            node_norm_first = []
            node_norm_matches = []
       
        if tmp_id == openai_final_keys[0]:
            yes_results_first[2] += 1
        else:
            no_results_first[2] += 1
        if tmp_id in node_norm_first:
            yes_results_first[0] += 1
        else:
            no_results_first[0] += 1
        if tmp_id == scibert_final_keys[0]:
            yes_results_first[1] += 1
        else:
            no_results_first[1] += 1

        """
        print("\n")
        print(tmp_id)
        print(openai_final_keys)
        print(node_norm_matches)
        print(scibert_final_keys)
        """
        oi_found = False
        nn_found = False

        #node norm, biobert, then openai
        if tmp_id in openai_final_keys:
            yes_results[2] += 1
            oi_found = True
            oi_index = openai_final_keys.index(tmp_id)
        else:
            no_results[2] += 1
        if tmp_id in node_norm_matches:
            yes_results[0] += 1
            nn_found = True
            for j, val in enumerate(node_norm_ids):
                if "UniProtKB:"+tmp_id in val:
                    nn_index = j
                    break
        else:
            no_results[0] += 1
        if tmp_id in scibert_final_keys:
            yes_results[1] += 1
        else:
            no_results[1] += 1
        
        #second step find identifier for node norm 
        if len(node_norm_keys) > 0 and nn_found:
            synonym_string = "\n".join(node_norm_keys)
            response = prompts.find_correct_identifier(entity, synonym_string)
            if response.lower() != "none":
                #print(response, node_norm_keys, "entity", entity)
                #print(index)
                try:
                    index = node_norm_keys.index(response)
                    if index-1 == nn_index:
                        yes_results_part_2[0] += 1
                    else:
                        no_results_part_2[0] += 1
                except: 
                    print(entity, uniprot_id)
                    print(response)
                    print('failed response nn')
            else:
                print(entity, uniprot_id)
                print(response)
                print('failed response nn')
         #second step find identifier for openai
        if oi_found:
            synonym_string = ""
            for i in range(len(openai_final_defs)):
                synonym_string += str(i+1) + ". " + openai_final_defs[i] + "\n"
            response = prompts.find_correct_identifier_openai(resp, synonym_string)
            if response.lower() != "none":
                try:
                    response = int(response)
                    if response-1 == oi_index:
                        yes_results_part_2[1] += 1
                    else:
                        no_results_part_2[1] += 1
                except:
                    print(entity, uniprot_id)
                    print(response)
                    print('failed response oi')
            else:
                print(entity, uniprot_id)
                print(response)
                print('failed response oi')
           
    categories = ["Node Norm", "OpenAI"]
    x = np.arange(len(categories))
    plt.clf()
    plt.close()
    plt.bar(x, yes_results_part_2, label='Yes', color=color_scheme.PRIMARY_COLOR, edgecolor=color_scheme.EDGE_COLOR)
    plt.bar(x, no_results_part_2, bottom=yes_results_part_2, label='No', color=color_scheme.SECONDARY_COLOR, edgecolor=color_scheme.EDGE_COLOR)
    plt.xticks(x, categories)
    plt.xlabel("Grounding Method Second Step")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, "stacked_barplot_second_step.png"), dpi=300)

           
    categories = ["Node Norm", "BioBERT", "OpenAI"]
    x = np.arange(len(categories))

    #check if the right answer appears in the top 10
    plt.clf()
    plt.close()
    plt.bar(x, yes_results, label='Yes', color=color_scheme.PRIMARY_COLOR, edgecolor=color_scheme.EDGE_COLOR)
    plt.bar(x, no_results, bottom=yes_results, label='No', color=color_scheme.SECONDARY_COLOR, edgecolor=color_scheme.EDGE_COLOR)
    plt.xticks(x, categories)
    plt.xlabel("Grounding Method")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, "stacked_barplot.png"), dpi=300)

    #check if the right answer is the top match
    plt.clf()
    plt.close()
    plt.bar(x, yes_results_first, label='Yes', color=color_scheme.PRIMARY_COLOR, edgecolor=color_scheme.EDGE_COLOR)
    plt.bar(x, no_results_first, bottom=yes_results_first, label='No', color=color_scheme.SECONDARY_COLOR, edgecolor=color_scheme.EDGE_COLOR)
    plt.xticks(x, categories)
    plt.xlabel("Counts")
    plt.ylabel("Grounding Method")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, "stacked_barplot_first_match.png"), dpi=300)


if __name__ == "__main__":
    main()
