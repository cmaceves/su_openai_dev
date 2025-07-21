import os
import sys
import json
import requests
import numpy as np
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()


def batch_request_formatting(entities, output_prefix, system_prompt):
    prefix = "context"
    requests = []
    for i, entity in enumerate(entities):
        request = {"custom_id": "context_%s"%str(i), "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "%s"%system_prompt},{"role": "user", "content": "%s"%entity}],"max_tokens": 200, "temperature":0}}
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
    if len(tmp) > 0:
        with open(os.path.join(output_prefix, "%s_%s.jsonl"%(prefix, counter)), "w") as jsonl_file:
            for value in tmp:
                jsonl_file.write(json.dumps(value) + "\n")


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

def retrieve_batch(basename, params_dir, output_dir):
    filename = os.path.join(output_dir, basename+"_params.txt")
    output_filename = os.path.join(output_dir, basename+"_results.json")

    #we have already retrieved this file
    #if os.path.isfile(output_filename):
    #    return

    with open(filename, "r") as ffile:
        data = json.load(ffile)
    output_file_id = data['batch_output_file_id']

    if output_file_id != "":
        #uses the output file id, which you can get using the batch id
        file_response = client.files.content(output_file_id)
        output_data = file_response.content
        with open(output_filename, 'wb') as jfile:
            jfile.write(output_data)
