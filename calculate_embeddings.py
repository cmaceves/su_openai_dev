"""
Given the uniprot database, please return the embedding vectors for each item.
"""
import os
import sys
import json
import numpy as np
from testing_grounding_strategies import save_array, calculate_tokens_per_chunk, create_batches, get_embeddings
import openai
from openai import OpenAI
from dotenv import load_dotenv
import prompts
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()


uniprot_db = "/home/caceves/su_openai_dev/parsed_databases/uniprot_definitions.json"
#with open(uniprot_db, "r") as jfile:
#    uniprot_def = json.load(jfile)

def process_embedding(input_dict, output_prefix):
    texts = []
    openai_records = []
    for i, record in enumerate(input_dict):
        if i  % 4000 == 0:
            print(i)
        resp = prompts.define_database_term(record['name'])
        texts.append(record['function'] + " " + resp)    
        tmp = {}
        tmp['accession'] = record['accession']
        tmp['name'] = record['name']
        tmp['function'] = record['function']
        tmp['openai'] = resp
        openai_records.append(tmp)

    with open("openai_%s_records.json"%output_prefix, "w") as jfile:
        json.dump(openai_records, jfile)

    token_counts = calculate_tokens_per_chunk(texts)
    batches = create_batches(token_counts)
    all_embeddings = []
    for i, batch in enumerate(batches):
        if i % 100 == 0:
            print("batch:", i)
        tmp = []
        for index in batch:
            tmp.append(texts[index])
        embeddings = get_embeddings(tmp)
        all_embeddings.extend(embeddings)

    all_embeddings = np.array(all_embeddings)
    save_array(all_embeddings, "openai_%s_embeddings.npy"%output_prefix)


def batch_request_formatting():
    database_name = "/home/caceves/su_openai_dev/parsed_databases/mesh_definitions.json"
    output_prefix = "/home/caceves/su_openai_dev/batch_request_formatted"
    system_prompt = "Given a biochemical entity describe it and define it's function in under 150 tokens"
    requests = []
    with open(database_name, 'r') as dfile:
        data = json.load(dfile)
    for record in data:
        request = {"custom_id": "%s"%record['accession'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "%s"%system_prompt},{"role": "user", "content": "%s"%record['name']}],"max_tokens": 200, "temperature":0}}
        requests.append(request)

    with open(os.path.join(output_prefix, "mesh.jsonl"), "w") as jsonl_file:
        for i, item in enumerate(requests):
            if i > 5:
                break
            jsonl_file.write(json.dumps(item) + "\n")

def upload_batch():
    #upload batch
    filename = "/home/caceves/su_openai_dev/batch_request_formatted/mesh.jsonl"
    batch_input_file = client.files.create(file=open(filename, "rb"), purpose="batch")

    #create batch
    batch_input_file_id = batch_input_file.id
    request_val = client.batches.create(input_file_id=batch_input_file_id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": "test job"})
    print(request_val.input_file_id)
    print(request_val.id)

def check_batch_status():
    batch = client.batches.retrieve("batch_67abea87d1a88190b737eddcf5dccc2e")
    print(batch)

def retrieve_batch():
    pass

def main():
    #uniprot and go are done
    #batch_request_formatting()
    #upload_batch()
    check_batch_status()
    """
    go_database = "/home/caceves/su_openai_dev/parsed_databases/go_definitions.json"
    with open(go_database, 'r') as gfile:
        go_def = json.load(gfile)
    process_embedding(go_def, "go")
    """
if __name__ == "__main__":
    main()
