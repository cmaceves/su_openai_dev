import os
import ast
import sys
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
from lxml import etree
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()
import nltk
nltk.download('punkt')
from joblib import Parallel, delayed
import prompts
from sklearn.metrics.pairwise import cosine_similarity
import evaluate_grounding_step

def parse_go_terms(filename):
    counter = 0
    all_strings = []
    all_ui = []
    all_names = [] 
    with open(filename, 'r') as ffile:
        for line in ffile:
            line = line.strip()
            line_list = line.split("\t")
            ui = line_list[0]
            name = line_list[1]
            definition = line_list[2]
            counter += 1
            pass_string = "Term: " + name + "\nDefinition: " + definition.lower()
            all_strings.append(str(pass_string))
            all_ui.append(ui)
            all_names.append(name)

    response = Parallel(n_jobs=40)(delayed(prompts.define_gpt)(value, i) for i, value in enumerate(all_strings[30000:]))
    gpt_def = {}
    for resp, name, ui in zip(response, all_names, all_ui[30000:]):
        gpt_def[ui] = resp
        
    with open('go_gpt_def_4.json', 'w') as jfile:
        json.dump(gpt_def, jfile)
   
def try_openai_embeddings(text, model="text-embedding-3-small"):
   embed = client.embeddings.create(input=text, model=model)
   vectors = []
   for x in embed.data:
       vec = x.embedding
       vectors.append(vec)
   return vectors

def main():
    """
    with open("new_master_def.json", 'r') as jfile:
        master_dict = json.load(jfile)

    embedding_vectors = np.load('new_openai_embedding.npy')
    print(embedding_vectors.shape)
    new_document = "Sodium-Chloride Symporter"
    resp = prompts.define_gpt(new_document.lower())
    vec = try_openai_embeddings(resp)
    vec = np.array(vec)
    vec = vec.reshape(1, -1)
    print("calcualting cosine scores")
    cosine_scores = cosine_similarity(vec, embedding_vectors)
    cosine_scores = cosine_scores.flatten()
    print("max cosine", max(cosine_scores))
    indexes = list(range(len(cosine_scores)))
    zipped = list(zip(cosine_scores, indexes))
    zipped.sort(reverse=True)
    cosine_scores, indexes = zip(*zipped)
    cosine_scores = list(cosine_scores)[:10]
    indexes = list(indexes)[:10]

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
    sys.exit(0)


    uniprot_json = "/home/caceves/su_openai_dev/databases/uniprot_human.json"
    uniprot_strings = []
    keys = []

    with open(uniprot_json, "r") as jfile:
        all_data = json.load(jfile)
        data = all_data['description']
        order = all_data['order']
    for key, value in data.items():
        tmp = ""
        keys.append(key)
        tmp += "Term: " + key + "\n" + "Definition: " + value
        uniprot_strings.append(tmp)
    print(len(uniprot_strings)) 
    uniprot_def_dict = {}
    response = Parallel(n_jobs=40)(delayed(prompts.define_gpt)(value, i) for i, value in enumerate(uniprot_strings))
    for resp, ui in zip(response, keys):
        uniprot_def_dict[ui] = resp
    with open('uniprot_def_1.json', 'w') as jfile:
        json.dump(uniprot_def_dict, jfile)
    sys.exit(0)
    
    master_dict = {}
    record_data = []
    labels = []

    uniprot_database = "/home/caceves/su_openai_dev/databases/uniprot_sprot.dat"
    counter = 0
    keep = False
    hold = []
    keep_dict = {}
    uniprot_order = []
    with open(uniprot_database, 'r') as ufile:
        for i,line in enumerate(ufile):
            if counter % 10000 == 0 and counter > 0:
                print(counter)
            line = line.strip()
            hold.append(line)
            if line == "//":
                if keep:
                    counter +=1
                    identifier = ""
                    description = ""
                    use = False
                    for h in hold:
                        if h.startswith("ID"):
                            h = h.split("   ")
                            uniprot_order.append(h[1])
                        elif h.startswith('CC'):
                            if "-!-" in h:
                                if "FUNCTION" in h:
                                    use = True
                                    h = h.split("-!- FUNCTION: ")[-1]
                                else: 
                                    use = False
                            if use:
                                h = h.split("       ")[-1]
                                description += h
                        elif h.startswith('DE') and "RecName" in h:
                            h = h.split("RecName: Full=")[-1]
                            identifier += h
                            
                    keep_dict[identifier] = description
                hold = []
            if line.startswith("ID"):
                if "human" in line.lower():
                    keep = True
                else:
                    keep = False
    with open("/home/caceves/su_openai_dev/databases/uniprot_human.json", "w") as jfile:
        json.dump({"description":keep_dict, "order":uniprot_order}, jfile)
    sys.exit(0)
    for go_filename in go_filenames:
        with open(go_filename, 'r') as gfile:
            data = json.load(gfile)
        for key, value in data.items():
            master_dict[key] = value
            record_data.append(value)
            labels.append(key)

    with open("master_def.json", "r") as gfile:
        data = json.load(gfile)
   
    for key, value in data.items():
        master_dict[key] = value
        record_data.append(value)
        labels.append(key)
    all_embed = []
    """
    
    """
    for i in range(0, len(record_data), 2000):
        print(i)
        vec = try_openai_embeddings(record_data[i:(i+2000)])
        all_embed.extend(vec)
    df = pd.DataFrame({"embedding":all_embed, "record":record_data})        
    print(df)
    df.to_csv("open_ai_embeddings.csv", index=False)
    """
    """
    df = pd.read_csv("open_ai_embeddings.csv")
    print("loaded embedding")
    #df['embedding'] = df['embedding'].apply(ast.literal_eval)
    print("ast applied")
    embedding_vectors = df['embedding'].tolist()
    new_vectors = []
    for i, vec in enumerate(embedding_vectors):
        if i % 1000 == 0 and i > 0:
            print(i)
        new_vectors.append(ast.literal_eval(vec))    
    embedding_vectors = np.array(new_vectors)
    np.save('openai_embedding.npy', embedding_vectors)
    sys.exit(0)
    """

    """
    embedding_vectors = np.load('openai_embedding.npy')
    print(embedding_vectors.shape)
    new_document = "lipid biosynthesis"
    resp = prompts.define_gpt(new_document)
    vec = try_openai_embeddings(resp)
    vec = np.array(vec)
    vec = vec.reshape(1, -1)
    print("calcualting cosine scores")
    cosine_scores = cosine_similarity(vec, embedding_vectors)
    cosine_scores = cosine_scores.flatten()
    print("max cosine", max(cosine_scores))
    indexes = list(range(len(cosine_scores)))
    zipped = list(zip(cosine_scores, indexes))
    zipped.sort(reverse=True)
    cosine_scores, indexes = zip(*zipped)
    cosine_scores = list(cosine_scores)[:10]
    indexes = list(indexes)[:10]
    for j, (key, value) in enumerate(master_dict.items()):
        if j in indexes:
            print(key, value)
    sys.exit(0)
    #go_filename = "./databases/go_definitions.txt"
    #parse_go_terms(go_filename)
    """
    mesh_filename = "mesh_db_2024.gz"
    tree = etree.parse(mesh_filename)
    root = tree.getroot()
   
    record_dictionary = {}
    all_names = []
    all_strings = []
    all_ui = []

    def find_all_tags(element):
        tags = set()
        for elem in element.iter():
            tags.add(elem.tag)
        return tags



    for i, record in enumerate(root.findall('DescriptorRecord')):
        if i % 5000 == 0:
            print(i)
        all_tags = find_all_tags(record)
        print(all_tags)
        tmp = record.find("DescriptorUI").text.strip()
        print(tmp)
        sys.exit(0)
        ui = record.find('DescriptorUI').text.strip()
        if ui == "MESH:C007088" or ui == "C007088":
            print(i, ui)
            sys.exit(0)
        name = record.find('DescriptorName').find("String").text.strip()
        concept_list = record.find('ConceptList')
        concept = concept_list.find('Concept')
        try:
           scope_note = concept.find('ScopeNote').text
        except:
            continue
        #record_data.append(name + " is " + scope_note.strip().lower())
        record_dictionary[ui] = name + " is " + scope_note.strip().lower()
        pass_string = "Term: " + name + "\nDefinition: " + scope_note.strip().lower()
        all_strings.append(str(pass_string))
        all_names.append(name)
        all_ui.append(ui)

    """
    response = Parallel(n_jobs=20)(delayed(prompts.define_gpt)(value, i) for i, value in enumerate(all_strings[25000:30000]))
    for resp, name, ui in zip(response, all_names, all_ui):
        gpt_def[ui] = resp
        #print(name, resp)
    with open('gpt_def_6.json', 'w') as jfile:
        json.dump(gpt_def, jfile)
    sys.exit(0)
    """
if __name__ == "__main__":
    main()
