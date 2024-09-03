import os
import sys
import json
import numpy as np

import prompts
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def try_openai_embeddings(text, model="text-embedding-3-small"):
   embed = client.embeddings.create(input=text, model=model)
   vectors = []
   for x in embed.data:
       vec = x.embedding
       vectors.append(vec)
   return vectors

def main(new_document, n=10):
    new_document = new_document.lower()
    with open("new_master_def.json", 'r') as jfile:
        master_dict = json.load(jfile)

    embedding_vectors = np.load('new_openai_embedding.npy')
    resp = prompts.define_gpt(new_document)
    vec = try_openai_embeddings(resp)
    vec = np.array(vec)
    vec = vec.reshape(1, -1)
    cosine_scores = cosine_similarity(vec, embedding_vectors)
    cosine_scores = cosine_scores.flatten()
    indexes = list(range(len(cosine_scores)))
    zipped = list(zip(cosine_scores, indexes))
    zipped.sort(reverse=True)
    cosine_scores, indexes = zip(*zipped)
    cosine_scores = list(cosine_scores)[:n]
    indexes = list(indexes)[:n]

    possible_matches = ""
    possible_matches_order = []

    counter = 1
    for j, (key, value) in enumerate(master_dict.items()):
        if j in indexes:
            possible_matches += str(counter) + ". " + value + "\n"
            possible_matches_order.append(key + " " + value)
            counter += 1
    resp, prompt = prompts.choose_embedding_match(possible_matches, new_document.lower())
    resp = int(resp)
    print(resp)
    print(possible_matches_order[resp-1])
