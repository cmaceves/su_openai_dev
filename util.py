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
from dotenv import load_dotenv
load_dotenv('/home/caceves/su_openai_dev/.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import prompts

def load_index_map(index_path='index_map.pkl'):
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def search_entity(search_value, n, merged_path='merged_embeddings.npy', index_path='index_map.pkl', real_vectors=None):
    # Load merged embeddings memmap and index map
    index_map = load_index_map(index_path)

    total_rows = sum(count for _, count in index_map)
    embedding_dim = None
    # Get dim from first file
    first_file = index_map[0][0]
    arr = np.load(first_file, mmap_mode='r')
    embedding_dim = arr.shape[1]

    embeddings = np.memmap(merged_path, dtype='float32', mode='r', shape=(total_rows, embedding_dim))

    # Get search embedding vector
    embedding_1 = np.array(get_embeddings([search_value]), dtype='float32')  # shape (1, embedding_dim)

    # Calculate distances all at once (you need to have a vectorized calculate_distances)
    distances = calculate_distances(embeddings, embedding_1)  # distances shape = (total_rows,)

    # Get top-n indices by smallest distance
    top_indices = np.argpartition(distances, n)[:n]
    top_indices = top_indices[np.argsort(distances[top_indices])]

    # Build reverse lookup (index -> (file, local_idx))
    reverse_map = []
    for file, count in index_map:
        reverse_map.extend([(file, i) for i in range(count)])

    # Extract results with file names and local indices
    return_names = []
    return_ids = []
    comparison_scores = None

    for idx in top_indices:
        file, local_idx = reverse_map[idx]
        def_file = file.replace('embeddings', 'batch_request_formatted')  # Adjust path if needed
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

    return return_names, return_ids, comparison_scores

def pubmed_search_json(term, max_results=10):
    # Define the base URLs for E-utilities API
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = base_url + "esearch.fcgi"
    fetch_url = base_url + "efetch.fcgi"

    # Step 1: Search for PubMed articles based on the search term
    max_results = 3000
    search_params = {
        "db": "pubmed",
        "term": term + " AND humans[MeSH Terms]",
        "retmax": max_results,  # Limit the number of results
        "retmode": "json"       # Return results in JSON format
    }
    search_response = requests.get(search_url, params=search_params)
    search_results = search_response.json()
    if len(search_results) == 0:
        return {}

    # Get the list of PubMed IDs (PMIDs)
    pmid_list = search_results["esearchresult"]["idlist"]
    if not pmid_list:
        return {}

    counter = 0
    max_counter = 100
    increment = 100
    all_titles = []
    while max_counter <= 500:
        pmid_list_tmp = pmid_list[counter:max_counter]
        counter += increment
        max_counter += increment
        # Step 2: Fetch abstracts for each PMID
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmid_list_tmp),  # Join PMIDs as a comma-separated string
            "retmode": "xml",          # Fetch results in JSON format
            "rettype": "abstract"
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        soup = BeautifulSoup(fetch_response.content, "xml")
        abstracts = {}
        titles = []
        for article in soup.find_all("PubmedArticle"):
            pmid = article.find("PMID").text
            abstract_text = ""
            titles.append(article.find("ArticleTitle").text)
            # Find all AbstractText sections and handle HTML tags within
            #for abstract_section in article.find_all("AbstractText"):
            #    abstract_text += abstract_section.get_text(" ", strip=True) + " "  # Join text with spaces

            # Store the cleaned abstract text by PMID
            #abstracts[pmid] = abstract_text.strip()
        all_titles.append(titles)
    return(abstracts, all_titles)

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
    #ingest_database()

    search_term = "Clocortolone pivalate"
    search_term = "phospholipase a2 prostaglandins"
    search_term = search_term.lower()
    print(search_term)
    abstracts = pubmed_search_json(search_term, max_results=10)
    for key, value in abstracts.items():
        entities, resp = prompts.extract_entities(value)
        print(key)
        print(entities)
        print("\n")

