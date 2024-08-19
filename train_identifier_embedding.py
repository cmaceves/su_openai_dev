import os
import sys
import json
import pandas as pd
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
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def main():
    go_filenames = ['go_gpt_def_1.json', 'go_gpt_def_2.json', 'go_gpt_def_3.json', 'go_gpt_def_4.json'] 
    master_dict = {}
    record_data = []
    
    for go_filename in go_filenames:
        with open(go_filename, 'r') as gfile:
            data = json.load(gfile)
        for key, value in data.items():
            master_dict[key] = value
            record_data.append(value)
    with open("master_def.json", "r") as gfile:
        data = json.load(gfile)
    
    for key, value in data.items():
        record_data.append(value)
    
    """
    all_embed = []
    for i, record in enumerate(record_data):
        vec = try_openai_embeddings(record)
        all_embed.append(vec)
    df = pd.DataFrame({"embedding":all_embed, "record":record_data})        
    df.to_csv("open_ai_embeddings.csv", index=False)
    sys.exit(0)
    """ 
    loaded_model = Doc2Vec.load("test_doc2vec_model.model")
    token = word_tokenize(new_document.lower())
    new_vector = loaded_model.infer_vector(token)
    print(len(record_data))
    similar_docs = loaded_model.dv.most_similar([new_vector], topn=10)
    for tag, similarity in similar_docs:
        print(f"Document tag: {tag}, Similarity: {similarity}") 
        print(record_data[int(tag)], "\n")
    sys.exit(0)
    
    #go_filename = "./databases/go_definitions.txt"
    #parse_go_terms(go_filename)
    mesh_filename = "mesh_db_2024.gz"
    tree = etree.parse(mesh_filename)
    root = tree.getroot()
   
    record_dictionary = {}
    all_names = []
    all_strings = []
    all_ui = []
    """
    for i, record in enumerate(root.findall('DescriptorRecord')):
        if i % 5000 == 0:
            print(i)
        ui = record.find('DescriptorUI').text.strip()
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
        #if ui == "D008148":
        #    print(i)
        #    sys.exit(0)
    response = Parallel(n_jobs=20)(delayed(prompts.define_gpt)(value, i) for i, value in enumerate(all_strings[25000:30000]))
    for resp, name, ui in zip(response, all_names, all_ui):
        gpt_def[ui] = resp
        #print(name, resp)
    with open('gpt_def_6.json', 'w') as jfile:
        json.dump(gpt_def, jfile)
    sys.exit(0)
    """ 
    print("tokenize documents")
    #preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                  tags=[str(i)]) for i,
                   doc in enumerate(record_data)]

    #train the Doc2vec model
    model = Doc2Vec(vector_size=100, min_count=1, epochs=50, seed=42, workers=8)
    print("building vocab")
    model.build_vocab(tagged_data)
    print("training model")
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    #save the model to a file
    model.save("test_doc2vec_model.model")
    sys.exit(0)

    new_document = "lovastatin"
    token = word_tokenize(new_document.lower())
    new_vector = model.infer_vector(token)
    print(len(record_data))

    similar_docs = model.dv.most_similar([new_vector], topn=10)
    for tag, similarity in similar_docs:
        print(f"Document tag: {tag}, Similarity: {similarity}") 
        print(record_data[int(tag)], "\n")

if __name__ == "__main__":
    main()
