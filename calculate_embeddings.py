"""
Given a database, please return the embedding vectors for each item.
"""
import os
import sys
import json
import requests
import numpy as np
from testing_grounding_strategies import save_array, calculate_tokens_per_chunk, create_batches, get_embeddings
import openai
from openai import OpenAI
from dotenv import load_dotenv
import prompts
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()

output_dir = "/home/caceves/su_openai_dev/batch_request_outputs"

def process_embedding(prefix):
    formatting_dir = "/home/caceves/su_openai_dev/embedding_request_formatted"

    #should already be in <50k chunks
    definition_filename = os.path.join(output_dir, prefix+"_results.json")

    text_definitions = []
    custom_ids = []
    with open(definition_filename, "r") as dfile:
        for line in dfile:
            data = json.loads(line)
            definition = data['response']['body']['choices'][0]['message']['content']
            text_definitions.append(definition)
            custom_ids.append(data['custom_id'])
    format_inputs = []
    for definition, custom_id in zip(text_definitions, custom_ids):
        tmp = {"custom_id": "%s"%custom_id, "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-ada-002", "input": "%s"%definition}}
        format_inputs.append(tmp)

    with open(os.path.join(formatting_dir, "%s_embedding.jsonl"%(prefix)), "w") as jsonl_file:
        for value in format_inputs:
            jsonl_file.write(json.dumps(value) + "\n")

    """
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
    """

def batch_request_formatting(prefix):
    database_name = "/home/caceves/su_openai_dev/parsed_databases/%s_definitions.json"%prefix
    output_prefix = "/home/caceves/su_openai_dev/batch_request_formatted"
    system_prompt = "Given a biochemical entity describe it and define it's function in under 150 tokens"
    requests = []
    with open(database_name, 'r') as dfile:
        data = json.load(dfile)
    for i, record in enumerate(data):
        request = {"custom_id": "%s"%record['accession'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "%s"%system_prompt},{"role": "user", "content": "%s"%record['name']}],"max_tokens": 200, "temperature":0}}
        requests.append(request)
    counter = 1
    tmp = []
    for i, item in enumerate(requests):
        if len(tmp) == 50000:
            with open(os.path.join(output_prefix, "%s_%s.jsonl"%(prefix, counter)), "w") as jsonl_file:
                for value in tmp:
                    jsonl_file.write(json.dumps(value) + "\n")
            counter += 1
            tmp = []
        tmp.append(item)

    #clean up anything that hasn't been written
    if len(tmp) > 0:
        with open(os.path.join(output_prefix, "%s_%s.jsonl"%(prefix, counter)), "w") as jsonl_file:
            for value in tmp:
                jsonl_file.write(json.dumps(value) + "\n")
   

def upload_embedding_batch(filename, prefix):
    """
    Upload a batch of pre-formatted requests to the OpenAI API embeddings.
    """
    #upload batch
    batch_input_file = client.files.create(file=open(filename, "rb"), purpose="batch")

    #create batch
    batch_input_file_id = batch_input_file.id
    request_val = client.batches.create(input_file_id=batch_input_file_id, endpoint="/v1/embeddings", completion_window="24h", metadata={"description": "%s"%prefix})
   
    output_dict = {"batch_input_file_id":request_val.input_file_id, "batch_output_file_id": "", "id":request_val.id, "status":"", "error_file_id":""}

    #write important information to a file
    output_file = os.path.join(output_dir, prefix+"_params.txt")
    with open(output_file, "w") as ofile:
        json.dump(output_dict, ofile)


def upload_batch(filename, prefix):
    """
    Upload a batch of pre-formatted requests to the OpenAI API.
    """
    #upload batch
    batch_input_file = client.files.create(file=open(filename, "rb"), purpose="batch")

    #create batch
    batch_input_file_id = batch_input_file.id
    request_val = client.batches.create(input_file_id=batch_input_file_id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": "%s"%prefix})
   
    output_dict = {"batch_input_file_id":request_val.input_file_id, "batch_output_file_id": "", "id":request_val.id, "status":"", "error_file_id":""}

    #write important information to a file
    output_file = os.path.join(output_dir, prefix+"_params.txt")
    with open(output_file, "w") as ofile:
        json.dump(output_dict, ofile)

def check_batch_status(basename):
    #defines the batch param filename
    filename = os.path.join(output_dir, basename+"_params.txt")
    with open(filename, "r") as ffile:
        data = json.load(ffile)
    batch_id = data['id']

    #uses the batch id
    batch = client.batches.retrieve(batch_id)
    if batch.status == "completed":
        data['batch_output_file_id'] = batch.output_file_id
        data['status'] = batch.status
        data['batch_input_file_id'] = batch.input_file_id
        with open(filename, 'w') as ffile:
            json.dump(data, ffile)
    
def retrieve_batch(basename):
    filename = os.path.join(output_dir, basename+"_params.txt")
    output_filename = os.path.join(output_dir, basename+"_results.json")

    #we have already retrieved this file
    if os.path.isfile(output_filename):
        return
    
    with open(filename, "r") as ffile:
        data = json.load(ffile)
    output_file_id = data['batch_output_file_id']

    if output_file_id != "":
        #uses the output file id, which you can get using the batch id
        file_response = client.files.content(output_file_id)
        output_data = file_response.content
        with open(output_filename, 'wb') as jfile:
            jfile.write(output_data)

def main():
    upload = True
    check = False

    embeddings = False
    embeddings_check = False

    #this will yield however many batches are present in the database
    database_name = "interpro"
    
    if upload:
        batch_request_formatting(database_name)
        #get all batches for this database
        format_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]
        #upload the batches
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            upload_batch(batch, basename)
    
    #check the batch status and if it's complete retrieve it
    if check:
        #get all batches for this database
        format_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            check_batch_status(basename)
            retrieve_batch(basename)

    if embeddings:
        #here we grab the actual embedding vector
        format_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]

        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            prefix = basename+"_vectors"

            #here we check if this batch has been successfully uploaded before
            output_file = os.path.join(output_dir, prefix + "_results.json")
            if os.path.isfile(output_file):
                continue

            process_embedding(basename)
            filename = os.path.join("/home/caceves/su_openai_dev/embedding_request_formatted", basename+"_embedding.jsonl")
            upload_embedding_batch(filename, prefix)

    if embeddings_check:
        format_dir = "/home/caceves/su_openai_dev/embedding_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]
        for batch in all_batches:
            basename = os.path.basename(batch).replace("_embedding.jsonl", "_vectors")
            check_batch_status(basename)
            retrieve_batch(basename)


if __name__ == "__main__":
    main()
