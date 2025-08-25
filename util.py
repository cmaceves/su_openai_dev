import os
import sys
import json
import pickle
import requests
import random
random.seed(42)
import numpy as np
import openai
from openai import OpenAI
from line_profiler import LineProfiler
from dotenv import load_dotenv
load_dotenv('/home/caceves/su_openai_dev/.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import prompts
from numba import njit, prange

def normalize_node(term, n=1):
    node_norm_url = "https://nodenorm.transltr.io/get_normalized_nodes?curie="
    resp = requests.get(node_norm_url+term)
    #print(node_norm_url+term)
    resp = resp.json()
    if term in resp:
        if resp[term] is None:
            return([])
        eq_id = [x['identifier'] for x in resp[term]['equivalent_identifiers']]
        return(eq_id)
    else:
        return([])

@njit(parallel=True, fastmath=True)
def cosine_distances_numba(embeddings, queries):
    """
    Compute cosine distances between all queries and all embeddings.

    Parameters
    ----------
    embeddings : np.ndarray of shape (N, D)
        Matrix of N embedding vectors of dimension D.
    queries : np.ndarray of shape (M, D)
        Matrix of M query vectors of dimension D.

    Returns
    -------
    distances : np.ndarray of shape (M, N)
        Cosine distances (1 - cosine similarity).
    """
    N, D = embeddings.shape
    M = queries.shape[0]
    distances = np.empty((M, N), dtype=np.float32)

    # precompute norms
    emb_norms = np.empty(N, dtype=np.float32)
    for i in prange(N):
        s = 0.0
        for d in range(D):
            s += embeddings[i, d] * embeddings[i, d]
        emb_norms[i] = np.sqrt(s)

    query_norms = np.empty(M, dtype=np.float32)
    for j in prange(M):
        s = 0.0
        for d in range(D):
            s += queries[j, d] * queries[j, d]
        query_norms[j] = np.sqrt(s)

    # compute cosine distances
    for j in prange(M):
        for i in range(N):
            dot = 0.0
            for d in range(D):
                dot += queries[j, d] * embeddings[i, d]
            sim = dot / (query_norms[j] * emb_norms[i] + 1e-9)
            distances[j, i] = 1.0 - sim

    return distances

def load_index_map(index_path='/home/caceves/su_openai_dev/index_map.pkl'):
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def cosine_distances_chunked(embeddings, query_vector, chunk_size=700000):
    """
    Compute cosine distances between embeddings and query_vector in chunks.
    embeddings: memmap or ndarray (N, D)
    query_vector: ndarray (1, D)
    """
    n = embeddings.shape[0]
    distances = np.empty(n, dtype=np.float32)
    q_norm_val = np.linalg.norm(query_vector)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]
        chunk_norms = np.sqrt(np.einsum('ij,ij->i', chunk, chunk))
        dot_products = np.dot(chunk, query_vector.T).squeeze()
        similarities = dot_products / (chunk_norms * q_norm_val)
        distances[start:end] = 1 - similarities
    return distances

def search_entity(search_value, n, merged_path='/home/caceves/su_openai_dev/merged_embeddings.npy', index_path='/home/caceves/su_openai_dev/index_map.pkl', real_vectors=None, reverse_map=None, embedding_dim=None, total_rows=None):
    # Load merged embeddings memmap and index map
    if reverse_map is None:
        index_map = load_index_map(index_path)
        reverse_map = []
        for file, count in index_map:
            reverse_map.extend([(file, i) for i in range(count)])
        total_rows = sum(count for _, count in index_map)
    if embedding_dim is None:
        first_file = index_map[0][0]
        arr = np.load(first_file, mmap_mode='r')
        embedding_dim = arr.shape[1]

    embeddings = np.memmap(merged_path, dtype='float32', mode='r', shape=(total_rows, embedding_dim))
    embedding_1 = np.array(get_embeddings([search_value]), dtype='float32')  # shape (1, embedding_dim)
    distances = cosine_distances_numba(embeddings, embedding_1)
    distances = distances.ravel()

    # Get top-n indices by smallest distance
    top_indices = np.argpartition(distances, n)[:n]
    top_indices = top_indices[np.argsort(distances[top_indices])]
    #distances = cosine_distances_chunked(embeddings, embedding_1)
    # Get top-n indices by smallest distance
    #top_indices = np.argpartition(distances, n)[:n]
    #top_indices = top_indices[np.argsort(distances[top_indices])]
    # Extract results with file names and local indices
    return_names = []
    return_ids = []
    comparison_scores = None
    for idx in top_indices:
        file, local_idx = reverse_map[idx]
        def_file = file.replace("/openai_","/").replace("_embeddings","")
        def_file = def_file.replace('embeddings', 'batch_request_formatted')  # Adjust path if needed
        def_file = def_file.replace('.npy', '.jsonl')
        if not os.path.isfile(def_file):
            continue
        with open(def_file, 'r') as dfile:
            lines = dfile.readlines()
            if local_idx >= len(lines):
                continue
            data = json.loads(lines[local_idx])
            id_val = data.get('custom_id')
            try:
                name = data['body']['messages'][1]['content']
            except Exception:
                name = None
            return_names.append(name)
            return_ids.append(id_val)
    # Optionally compare to real vectors if provided
    if real_vectors is not None:
        # Pick best match
        best_idx = top_indices[0]
        tmp = embeddings[best_idx].reshape(1, -1)
        real_vectors = real_vectors.reshape(1, -1)
        dist = calculate_distances(real_vectors, tmp)
        comparison_scores = dist[0]

    top_embedding = embeddings[top_indices[0]].copy()

    return return_names, return_ids, comparison_scores, top_embedding

def save_array(array, filename):
    np.save(filename, array)

def load_array(filename):
    return np.load(filename, mmap_mode='r')

def get_embeddings(texts, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=texts, model=model)
    embeddings = [res.embedding for res in response.data]
    return embeddings

# Calculate distances between target vector and all vectors in the array
def calculate_distances(array, target_vector, metric='euclidean'):
    distances = cdist(array, target_vector, metric=metric)
    return distances.flatten()

def parse_indication_paths(indication_json, n=None):
    with open(indication_json, 'r') as jfile:
        data = json.load(jfile)
    if n is not None:
        selected_items = random.choices(data, k=n)
    else:
        selected_items = data

    return(selected_items)

def try_openai_embeddings(text, model="text-embedding-3-small"):
    """
    Queries OpenAI for vector embeddings.
    """
    embed = client.embeddings.create(input=text, model=model)
    vectors = []
    for x in embed.data:
        vec = x.embedding
        vectors.append(vec)
    return vectors

def ingest_database():
    """
    Code that isn't very generalizable for ingesting databases.
    """
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
    main()
