import os
import sys
import json
import random
random.seed(42)
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()

from joblib import Parallel, delayed
import prompts

def parse_indication_paths(indication_json, n=None):
    """
    indication_json : string
        The full path to the file containing the known indiciation paths.
    n : int
        The number of indiciations to randomly choose for evaluation.

    Parses the json file containing the indications and randomly selects N for evaluation.
    """
    with open(indication_json, 'r') as jfile:
        data = json.load(jfile)
    if n is not None:
        selected_items = random.choices(data, k=n)
    else:
        selected_items = data

    return(selected_items)

def try_openai_embeddings(text, model="text-embedding-3-small"):
   embed = client.embeddings.create(input=text, model=model)
   vectors = []
   for x in embed.data:
       vec = x.embedding
       vectors.append(vec)
   return vectors

def ingest_database():
    chebi_database = "/home/caceves/su_openai_dev/databases/ChEBI_complete_3star.sdf"
    hpo_database = "/home/caceves/su_openai_dev/databases/hp.json"

    with open(hpo_database, 'r') as hfile:
        data = json.load(hfile)
    print(data['graphs'][0]['nodes'])

    """
    #CHEBI
    with open(database_location, "r") as dfile:
        saved_lines = []
        tmp = ""
        all_ids = []
        all_strings = []
        for count, line in enumerate(dfile):
            if count == 0:
                continue
            line = line.strip()
            saved_lines.append(line)
            past_line = saved_lines[count-2]
            if "M  END" in past_line and count > 100:
                all_strings.append(tmp)
                tmp = ""
            if "<ChEBI Name>" in past_line:
                tmp += "Term:" + line + "\n"
            if "<Definition>" in past_line:
                tmp += "Definition:" + line
            if "<ChEBI ID>" in past_line:
                chebi_id = line
                all_ids.append(chebi_id)
    """

    

    sys.exit(0)
    response = Parallel(n_jobs=20)(delayed(prompts.define_gpt)(value, i) for i, (value) in enumerate(all_strings))
    all_dict = {}

    with open("master_dict_2.json", "r") as jfile:
        all_dict = json.load(jfile)
    for resp, identifier in zip(response, all_ids):
        all_dict[identifier] = resp

    vectors = try_openai_embeddings(response) 
    new_vectors = np.array(vectors)
    print(new_vectors.shape)
    embedding_vectors = np.load('new_openai_embedding.npy')
    print(embedding_vectors.shape)
    final_vectors = np.vstack((embedding_vectors, new_vectors))
    print(final_vectors.shape)
    print(len(all_dict))

    np.save('openai_embedding_2.npy', final_vectors)
    with open("master_dict_3.json", "w") as jfile:
        json.dump(all_dict, jfile)

if __name__ == "__main__":
    ingest_database()

