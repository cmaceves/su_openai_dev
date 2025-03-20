"""
Given a database, please return the embedding vectors for each item.
"""
import os
import sys
import json
import requests
import numpy as np
from testing_grounding_strategies import save_array, create_batches, get_embeddings
import openai
from openai import OpenAI
from dotenv import load_dotenv
import prompts
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()


def process_embedding(prefix, output_dir):
    formatting_dir = "/home/caceves/su_openai_dev/embedding_request_formatted"
    original_parsed_data = "/home/caceves/su_openai_dev/parsed_databases/%s_definitions.json"

    #should already be in <50k chunks
    definition_filename = os.path.join(output_dir, prefix+"_results.json")
    print("need to include original definition")
    sys.exit(0)
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

def batch_request_formatting(prefix, database_name, output_prefix, system_prompt, description=False):
    """
    prefix : prefix of the database being processed
    database_name : the full path to the database to process
    output_prefix : the directory to save the formatted batch requests to
    system_prompt : the system prompt
    description : do we include the entity description in the request or not
    """
    requests = []
    with open(database_name, 'r') as dfile:
        data = json.load(dfile)
    for i, record in enumerate(data):
        if not description:
            request = {"custom_id": "%s"%record['accession'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "%s"%system_prompt},{"role": "user", "content": "%s"%record['name']}],"max_tokens": 200, "temperature":0}}
        else:
            request = {"custom_id": "UNIPROT:%s"%record['accession'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "%s"%system_prompt},{"role": "user", "content": "%s - %s"%(record['name'], record['description'])}],"max_tokens": 200, "temperature":0}}
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
   

def upload_embedding_batch(filename, prefix, output_dir):
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


def upload_batch(filename, prefix, output_dir):
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

def check_batch_status(basename, output_dir):
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
    
def retrieve_batch(basename, output_dir):
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
    upload = False
    check = False

    embeddings = False
    embeddings_check = True

    #this will yield however many batches are present in the database
    database_name = "uniprot"
    output_dir = "/home/caceves/su_openai_dev/batch_request_outputs"

    if upload:
        database_full = "/home/caceves/su_openai_dev/parsed_databases/%s_definitions.json"%database_name
        output_prefix = "/home/caceves/su_openai_dev/batch_request_formatted"
        system_prompt = "Given a biochemical entity describe it and define it's function in under 150 tokens"
        
        batch_request_formatting(database_name, database_full, output_prefix, system_prompt)

        #get all batches for this database
        format_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]
        #upload the batches
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            upload_batch(batch, basename, output_dir)
    
    #check the batch status and if it's complete retrieve it
    if check:
        #get all batches for this database
        format_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            check_batch_status(basename, output_dir)
            retrieve_batch(basename, output_dir)

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

            process_embedding(basename, output_dir)
            filename = os.path.join("/home/caceves/su_openai_dev/embedding_request_formatted", basename+"_embedding.jsonl")
            upload_embedding_batch(filename, prefix, output_dir)

    if embeddings_check:
        format_dir = "/home/caceves/su_openai_dev/embedding_request_formatted"
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database_name)]
        for batch in all_batches:
            basename = os.path.basename(batch).replace("_embedding.jsonl", "_vectors")
            check_batch_status(basename, output_dir)
            retrieve_batch(basename, output_dir)


if __name__ == "__main__":
    main()
