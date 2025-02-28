import os
import sys
import json
import numpy as np
import util
from testing_grounding_strategies import load_array, get_embeddings, fetch_openai_scores

mesh_database = "/home/caceves/su_openai_dev/embeddings/openai_mesh_1_embeddings.npy"
mesh_supporting_file = "/home/caceves/su_openai_dev/batch_request_outputs/mesh_1_results.json"
mesh_definitions = "/home/caceves/su_openai_dev/parsed_databases/mesh_definitions.json"

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

with open(mesh_definitions, 'r') as mfile:
    mesh_data = json.load(mfile)

term_ids = []
term_definitions = []
with open(mesh_supporting_file, 'r') as mfile:
    for line in mfile:
        data = json.loads(line)
        term_id = data['custom_id']
        term_ids.append(term_id)
        term_def = data['response']['body']['choices'][0]['message']['content']
        term_definitions.append(term_def)

openai_embedding = load_array(mesh_database)

for i, (drug, diseases) in enumerate(all_disease_dict.items()):
    if i == 0:
        continue
    keys, defs = fetch_openai_scores(drug, openai_embedding, term_ids, term_definitions, 500)
    print(drug)
    print(diseases)
    for key, definition in zip(keys, defs):
        for value in mesh_data:
            if value['accession'] == key:
                name = value['name']
                break

        print(key, name)
    sys.exit(0)

