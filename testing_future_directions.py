import os
import sys
import json
import numpy as np
import util
from util import load_array, save_array, get_embeddings
from testing_grounding_strategies import fetch_openai_scores

embedding_dir  = "/home/caceves/su_openai_dev/embeddings"
supporting_file_dir = "/home/caceves/su_openai_dev/batch_request_outputs"
definition_dir = "/home/caceves/su_openai_dev/parsed_databases"
type_dir = "/home/caceves/su_openai_dev/typed_nodes"
def parse_diseases_of_drugs(indication_json, drugs):
    """
    For every drug return all associated diseases.
    """
    drug_disease_dict = {}
    with open(indication_json, 'r') as ifile:
        data = json.load(ifile)
    for value in data:
        tmp_drug = value['graph']['drug'].lower()
        tmp_disease = value['graph']['disease']
        if tmp_drug in drugs:
            if tmp_drug not in drug_disease_dict:
                drug_disease_dict[tmp_drug] = []
            drug_disease_dict[tmp_drug].append(tmp_disease)

    return(drug_disease_dict)

#parse the indications from drug central
indication_json = "indication_paths.json"
if not os.path.exists(indication_json):
    print("Downloading indications file...")
    indication_url = "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.json"
    r = requests.get(indication_url, allow_redirects=True)
    open(indication_json, 'wb').write(r.content)

evaluation_indications = util.parse_indication_paths(indication_json, 50)

#find 50 unique drugs
drugs = [x['graph']['drug'].lower() for x in evaluation_indications]
drugs = list(np.unique(drugs))

#here we get all diseases associated with the drug
all_disease_dict = parse_diseases_of_drugs(indication_json, drugs)

#define the embeddings you want to compare to
databases = ['ncbitaxon', 'mesh']
all_embedding_files = []
for database in databases:
    all_embedding_files.extend([os.path.join(embedding_dir, x) for x in os.listdir(embedding_dir) if database in x])

all_supporting_files = []
for filename in all_embedding_files:
    tmp = os.path.basename(filename).replace("openai_","").replace("_embeddings.npy", "_results.json")
    tmp = os.path.join(supporting_file_dir, tmp) 
    all_supporting_files.append(tmp)

all_name_files = []
for filename in all_embedding_files:
    database = os.path.basename(filename).split("_")[1]
    tmp = os.path.join(definition_dir, database + "_definitions.json") 
    all_name_files.append(tmp)

all_type_files = []
for filename in all_embedding_files:
    database = "_".join(os.path.basename(filename).split("_")[1:3])
    tmp = os.path.join(type_dir, database + ".json") 
    all_type_files.append(tmp)

for i, (drug, diseases) in enumerate(all_disease_dict.items()):
    if i < 1:
        continue
    drug = "lovastatin"
    for embedding_file, supporting_file, name_file, type_file in zip(all_embedding_files, all_supporting_files, all_name_files, all_type_files):
        term_ids = []
        term_definitions = []
        if "mesh" not in name_file:
            continue
        with open(supporting_file, 'r') as mfile:
            for line in mfile:
                data = json.loads(line)
                term_id = data['custom_id']
                term_ids.append(term_id)
                term_def = data['response']['body']['choices'][0]['message']['content']
                term_definitions.append(term_def)
        with open(name_file, 'r') as nfile:
            name_data = json.load(nfile)

        with open(type_file, 'r') as tfile:
            type_data = json.load(tfile)
        openai_embedding = load_array(embedding_file)
        keys, defs = fetch_openai_scores(drug, openai_embedding, term_ids, term_definitions, 1000)
        print(drug)
        print(diseases)
        for key, definition in zip(keys, defs):
            for value in name_data:
                if value['accession'] == key:
                    name = value['name']
                    break
            if type_data[key] == "Disease":        
                print(name), #type_data[key])
    sys.exit(0)



