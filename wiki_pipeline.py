import re
import os
import sys
import ast
import copy
from bs4 import BeautifulSoup
import bioc
import json
import pickle
import requests
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from openai import OpenAI
from normalize import get_normalizer #taken from Benchmarks
import pull_down_literature
import prompts
from wikidata.client import Client
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
categories = ["protien", "receptor", "enzyme", "cell", "disease", "gene family", "endogenous small molecule"]


def grounded_type(grounded, node_norm_url = "https://nodenorm.transltr.io/get_normalized_nodes?curie="):
    grounded_type = {}
    for key, value in grounded.items():
        if key not in grounded_type:
            grounded_type[key] = []
        for synonym, identifier_list in value.items():
            #print(key, value, synonym, identifier)
            for i, identifier in enumerate(identifier_list):
                if identifier == "":
                    continue
                if i == 0:
                    prefix = "MESH:"
                elif i == 1:
                    prefix = "UniprotKB:"
                elif i == 2:
                    prefix = "MONDO:"
                elif i == 3:
                    prefix == "HPO:"
                #mesh_id, uniprot, mondo, hpo order for grounding storage
                identifier = prefix + identifier[0]
                resp = requests.get(node_norm_url + identifier)
                data = resp.json()[identifier]
                if data is None:
                    continue
                data_type = data['type'][0]
                grounded_type[key].append(data_type)

    return(grounded_type)

def ground_synonym(term):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
    "action": "parse",
    "format": "json",
    "page": "%s" %(term)
    }
    resp = requests.get(url=url, params=params)
    data = resp.json()
    html = data['parse']['text']['*']
    soup = BeautifulSoup(html, "html.parser")
    redirect = False
    redirect_term = ""
    wikibase_id = ""
    #print(bcolors.OKBLUE + term)
    for i, paragraph in enumerate(soup.find_all("p")):
        if paragraph.text.strip() == "Redirect to:":
            for a in soup.find_all("a"):
                #print(bcolors.OKCYAN + str(paragraph))
                #print(bcolors.OKCYAN + str(a))
                redirect_term = a.text.strip()
                redirect = True
                break
    if redirect:
        params = {
        "action": "parse",
        "format": "json",
        "page": "%s" %(redirect_term)
        }
        resp = requests.get(url=url, params=params)
        data = resp.json()
    if "error" in data.keys():
        print(data)
        print(term, "bool redirect", redirect, "redirect", redirect_term)
        sys.exit(1)

    for value in data['parse']['properties']:
        if value['name'] == 'wikibase_item':
            wikibase_id = value['*']
    mesh_id = ""
    uniprot = ""
    drugbank = "" #not written in yet
    mondo = ""
    hpo = ""
    if wikibase_id == "":
        return("", "", "", "")
    wiki_client = Client() 
    wiki_entity = wiki_client.get(wikibase_id, load=True)
    for x in wiki_entity.keys():
        child = wiki_client.get(x.id)
        label = str(child.label).strip()
        if label == "MeSH descriptor ID":
            mesh_id = wiki_entity.getlist(child)
        if label == "UniProt protein ID":
            uniprot = wiki_entity.getlist(child)
        if label == "Mondo ID":
            mondo = wiki_entity.getlist(child)
        if label == "Human Phenotype Ontology ID":
            hpo = wiki_entity.getlist(child)
    return(mesh_id, uniprot, mondo, hpo)


def get_additional_text(expansion_list, term1, term2, mechanism):
    """
    Given a list of wikipedia page titles with information possibily relevant, query wikipedia for the page information and then summarize the page.
    """
    expansion_list = list(np.unique(expansion_list))
    all_additional_text =""
    for expansion in expansion_list:
        response, redirect, wikibase_id, linked_entities = query_wikipedia(expansion) 
        if redirect:
            response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
        response = response.replace("\n","")
        additional_information = summarize(response, term1, term2, mechanism)
        all_additional_text += additional_information
        #print(bcolors.OKGREEN + expansion, bcolors.OKCYAN + additional_information)       

    return(all_additional_text)

def matched_wikipedia_pages(entity_list):
    useful_pages = []
    for entity in entity_list:
        url = "https://en.wikipedia.org/w/api.php"
        text_all = ""
        wikibase_id = ""
        params = {
        "action": "opensearch",
        "format": "json",
        "search": "%s" %(entity)
        }
        resp = requests.get(url=url, params=params)
        loose_data = resp.json()
        useful = []
        if len(loose_data) < 2:
            continue
        try:
            ll = loose_data[1]
        except:
            print(entity)
            print("failure", loose_data)
            sys.exit(1)
        for possible_match in loose_data[1]:
            if possible_match.lower() == entity.lower():
                useful_pages.append(possible_match)
    return(useful_pages)


def corresponding_pages(entity_list, mechanism):
    """
    Given a list of entities, search wikipedia loosely for possibly matching pages. If the page title could possibly be relevant to the mechanism, return the page title.
    """
    useful_pages = []
    for entity in entity_list:
        url = "https://en.wikipedia.org/w/api.php"
        text_all = ""
        wikibase_id = ""
        params = {
        "action": "opensearch",
        "format": "json",
        "search": "%s" %(entity)
        }
        resp = requests.get(url=url, params=params)
        loose_data = resp.json()
        useful = []
        for possible_match in loose_data[1]:
            resp, prompt = prompts.test_prompt(possible_match, mechanism)
            if resp.lower() == "yes":
                useful.append(possible_match)
        useful_pages.extend(useful)
    return(useful_pages)

def expand_entities(entity_list):
    all_entities = []
    for entity in entity_list:
        sub, prompt  = prompts.clean_up_entities(entity)
        sub_entities = sub.split(",")
        sub_entities = [re.sub(r'^\s+', '', text) for text in sub_entities]
        sub_entities = [remove_punctuation(text) for text in sub_entities]
        all_entities.append(entity)
        all_entities.extend(sub_entities)
    all_entities = list(np.unique(all_entities))
    return(all_entities)

def remove_punctuation(input_string):
    # Pattern to match all punctuation
    pattern = r'[^\w\s]'
    # Replace all punctuation with an empty string
    return re.sub(pattern, '', input_string)

def summarize(text, drug, disease, mechanism):
    token_limit = 10000
    sentence_list = text.split(".")
    chunks = []
    tmp = ""
    tokens = 0
    for sentence in sentence_list:
        tokens += len(sentence.split(" "))
        tmp += sentence + "."
        if tokens > token_limit:
            chunks.append(tmp)
            tmp = ""
            tokens = 0
    chunks.append(tmp)
    add_info = ""
    for chunk in chunks:
        resp, prompt = prompts.summarize_add_info(chunk, mechanism)
        add_info += resp + " "
    return(add_info)

def get_wikidata_id(wikibase_id, lang='en'):
    wiki_client = Client() 
    wiki_entity = wiki_client.get(wikibase_id, load=True)
    for x in wiki_entity.keys():
        child = wiki_client.get(x.id)
        label = str(child.label).strip()
        print(wiki_entity.getlist(child))
        if label == "MeSH descriptor ID":
            mesh_id = wiki_entity.getlist(child)
        if label == "UniProt protein ID":
            uniprot = wiki_entity.getlist(child)
        if label == "Mondo ID":
            mondo = wiki_entity.getlist(child)
        if label == "Human Phenotype Ontology ID":
            hpo = wiki_entity.getlist(child)
        #print(term, label, wiki_entity.getlist(child))


def alternate_path(term1, term2, predicate_data):
    wiki_client = Client() 
    response, redirect, wikibase_id, linked_entities = query_wikipedia(term1)
    linked_entities = [x.lower() for x in linked_entities]
 
    if redirect:
        response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
    if wikibase_id != "":
        wiki_entity = wiki_client.get(wikibase_id, load=True)
        for x in wiki_entity.keys():
            child = wiki_client.get(x.id)
            label = str(child.label).strip()
            if label == "MeSH descriptor ID":
                mesh_id = wiki_entity.getlist(child)
    #get_wikidata_id(wikibase_id)

    linked_entities_1 = [x.lower() for x in linked_entities]        
    tmp = response.split(".")
    paragraph1 = ""
    for sentence in tmp:
        paragraph1 += sentence.strip() + ". "
    all_entities = linked_entities_1
    response, redirect, wikibase_id, linked_entities = query_wikipedia(term2)
    if redirect:
        response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
    linked_entities_2 = [x.lower() for x in linked_entities]
    overlap = [x for x in linked_entities_1 if x in linked_entities_2]
    overlap = list(np.unique(overlap))
    tmp = response.split(".")
    paragraph2 = ""
    for sentence in tmp:
        paragraph2 += sentence.strip() + ". "
    
    paragraph = paragraph1 + " " + paragraph2

    try:
        response, prompt = prompts.extract_mech_path(paragraph, term1, term2)
    except:
        return({})
    response_list = response.split("\n")
    mechanism = response_list[0].replace("Mechanism: ", "")
    entities = response_list[-1].replace("Relevant Entities: ","")
    entity_list = entities.split(", ")
    entity_list = [x.lower().replace(".","") for x in entity_list]

    synonym_dict = {}
    synonym_pages_dict = {}
    for entity in entity_list:
        resp, prompt = prompts.synonym_prompt(entity)
        synonym_list = resp.split("\n")
        final_synonym_list = []
        for synonym in synonym_list:
            resp, prompt = prompts.synonym_context(entity, synonym, mechanism)
            if resp.lower() != "yes":
                continue
            final_synonym_list.append(synonym)
            print(bcolors.HEADER+resp, bcolors.OKBLUE+entity, bcolors.OKGREEN+synonym)
        synonym_dict[entity] = final_synonym_list
        #print(bcolors.OKBLUE + "entity " + entity)
        #print(bcolors.OKCYAN + str(synonym_list))
        synonym_pages = matched_wikipedia_pages(final_synonym_list)
        synonym_pages_dict[entity] = synonym_pages
    print(response)
    synonym_groundings = {}
    for key, value in synonym_pages_dict.items():
        if key not in synonym_groundings:
            synonym_groundings[key] = {}
        for entity in value:
            if entity in synonym_groundings[key]:
                continue
            mesh_id, uniprot, mondo, hpo = ground_synonym(entity) 
            print(bcolors.OKBLUE + key, bcolors.OKCYAN + entity)
            print(bcolors.HEADER + str(mesh_id), uniprot, mondo, hpo)
            synonym_groundings[key][entity] = [mesh_id, uniprot, mondo, hpo]
    grounded_types = grounded_type(synonym_groundings) 
    with open("test_json.json", "w") as jfile:
        json.dump({"grounded":synonym_groundings, "grounded_types": grounded_types, "entities":entities, "mechansim":mechanism, "text":paragraph}, jfile)
    sys.exit(0)

    #additional information
    useful_expansions = corresponding_pages(entity_list, mechanism)
    print(bcolors.OKCYAN + "Useful supporting wikipedia pages")
    print(bcolors.OKCYAN + str(useful_expansions))
    additional_text = get_additional_text(useful_expansions, term1, term2, mechanism)
    print(additional_text)

    #re-extract the mechanism of action
    response, prompt = prompts.extract_mech_path(additional_text , term1, term2)
    print(bcolors.HEADER + response)

    sys.exit(0)

    entity_string = ""
    grounded_ids = {}
    all_supporting_information = ""
    print(bcolors.HEADER + str(entity_list))
    #this expands to provide additional supporting information
    for i, entity in enumerate(entity_list):
        if i != 0:
            entity_string += "\n"
        entity_string += entity
        #print("\n", bcolors.OKBLUE + entity + bcolors.OKBLUE) 
        text_all = loose_query_wikipedia(entity)
        t = ""
        for key, value in text_all.items():
            
            t += value + " "
        print(entity)
        print(t)
        additional_information = summarize(t, term1, term2, mechanism)
        all_supporting_information += additional_information
        #print(bcolors.OKCYAN + "additional info " + bcolors.OKCYAN , bcolors.OKCYAN + additional_information + bcolors.OKCYAN)
    sys.exit(0)
    #print(bcolors.HEADER + all_supporting_information + bcolors.HEADER)
    response, prompt = prompts.extract_mech_path(all_supporting_information, term1, term2)
    response_list = response.split("\n")
    final_mech = response_list[0].replace("Mechanism: ", "")
    final_entities = response_list[-1].replace("Relevant Entities: ","")
    final_entity_list = final_entities.split(", ")
    final_entity_list = [x.lower() for x in final_entity_list]
    final_entity_list = expand_entities(final_entity_list)
    print(final_entity_list)
    print(bcolors.OKBLUE + response)
    sys.exit(0)
    #final_entity_list = entity_list
    #ground using pubtator?
    #ground_pubtator(final_entity_list)


    for i, entity in enumerate(final_entity_list):
        if entity not in grounded_ids:
            grounded_ids[entity] = query_wikipedia_mesh(entity) 
    print(bcolors.OKGREEN + "response", bcolors.OKGREEN + response)
    print(grounded_ids)
    sys.exit(0)
    first_check = {"mechanism":mechanism, "entities":entity_list, "grounded":grounded_ids, "all_supporting_info":all_supporting_information, "final_mech":final_mech, "final_entities":final_entities}
    print(bcolors.OKBLUE + str(grounded_ids))
    return(first_check) 
    

def query_wikipedia_mesh(term):
    """
    This bit of code literally just tries to use wikipedia to ground.
    """
    url = "https://en.wikipedia.org/w/api.php"
    text_all = ""
    wikibase_id = ""
    params = {
    "action": "opensearch",
    "format": "json",
    "search": "%s" %(term)
    }
    resp = requests.get(url=url, params=params)
    data = resp.json()
    
    for possible_match in data[1]:
        print(bcolors.OKGREEN + "mesh term", term, bcolors.OKGREEN + str(possible_match) + bcolors.OKGREEN)
        print(lemmatizer.lemmatize(remove_punctuation(term.lower())), lemmatizer.lemmatize(remove_punctuation(possible_match.lower()))) 
        if lemmatizer.lemmatize(remove_punctuation(term.lower())) == lemmatizer.lemmatize(remove_punctuation(possible_match.lower())):
            term = possible_match
            break

    #print("found term ", term)
    params = {
    "action": "parse",
    "format": "json",
    "page": "%s" %(term)
    }
    resp = requests.get(url=url, params=params)
    data = resp.json()
    if 'error' in data:
        print("error", data, term)
        return("", "", "", "")
    for value in data['parse']['properties']:
        if value['name'] == 'wikibase_item':
            wikibase_id = value['*']
    mesh_id = ""
    uniprot = ""
    drugbank = "" #not written in yet
    mondo = ""
    hpo = ""
    if wikibase_id != "":
        wiki_client = Client() 
        wiki_entity = wiki_client.get(wikibase_id, load=True)
        for x in wiki_entity.keys():
            child = wiki_client.get(x.id)
            label = str(child.label).strip()
            if label == "MeSH descriptor ID":
                mesh_id = wiki_entity.getlist(child)
            if label == "UniProt protein ID":
                uniprot = wiki_entity.getlist(child)
            if label == "Mondo ID":
                mondo = wiki_entity.getlist(child)
            if label == "Human Phenotype Ontology ID":
                hpo = wiki_entity.getlist(child)
            #print(term, label, wiki_entity.getlist(child))
    return(mesh_id, uniprot, mondo, hpo)

def loose_query_wikipedia(term):
    url = "https://en.wikipedia.org/w/api.php"
    text_all = ""
    wikibase_id = ""
    params = {
    "action": "opensearch",
    "format": "json",
    "search": "%s" %(term)
    }
    resp = requests.get(url=url, params=params)
    loose_data = resp.json()
    text_all = {}
    for possible_match in loose_data[1]:
        #print("possible search match ", possible_match)  
        text_all[possible_match] = ""
        params = {
        "action": "parse",
        "format": "json",
        "page": "%s" %(possible_match)
        }
        resp = requests.get(url=url, params=params)
        data = resp.json()
        html = data["parse"]["text"]["*"]
        for value in data['parse']['properties']:
            if value['name'] == 'wikibase_item':
                wikibase_id = value['*']
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select('p a[href]'):
            try: 
                title = tag['title']
            except:
                continue
        for i, paragraph in enumerate(soup.find_all("p")):
            if i > 0:
                text_all[possible_match] += "\n"
            text_all[possible_match] += paragraph.text.strip()
    return(text_all)

def query_wikipedia(term):
    url = "https://en.wikipedia.org/w/api.php"
    text_all = ""
    wikibase_id = ""
    params = {
    "action": "parse",
    "page": "%s" %(term),
    "format": "json"
    }
    resp = requests.get(url=url, params=params)
    data = resp.json()
    if 'error' in data:
        return("", False, "", [])
    html = data["parse"]["text"]["*"]
    for value in data['parse']['properties']:
        if value['name'] == 'wikibase_item':
            wikibase_id = value['*']
    soup = BeautifulSoup(html, "html.parser")
    text_all = ""
    redirect_term = ""
    linked_entities = []
    for tag in soup.select('p a[href]'):
        try: 
            title = tag['title']
            linked_entities.append(title)
        except:
            continue
    for i, paragraph in enumerate(soup.find_all("p")):
        if i > 0:
            text_all += "\n"
        if paragraph.text.strip() == "Redirect to:":
            for a in soup.find_all("a"):
                redirect_term = a.text.strip()
        text_all += paragraph.text.strip()
    if redirect_term == "":
        return(text_all, False, wikibase_id, linked_entities)
    else:
        return(redirect_term, True, wikibase_id, linked_entities)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_solutions(json_file, solution_df):
    all_names = []
    with open(json_file, 'r') as jfile:
        data = json.load(jfile)
    for index, row in solution_df.iterrows():
        solution_steps = solution_df.iloc[index].tolist()
        found = False
        for graph in data:
            node_list = graph['nodes']
            ids = []
            names = []
            for nl in node_list:
                ids.append(nl['id'])
                names.append(nl['name'])
            #if "MESH:D012148" in ids and "MESH:D012148" in ids:
            #    print("here", ids, names)
            if len(ids) != len(solution_steps):
                continue
            if ids == solution_steps:
                found = True
                all_names.append(names)
        if found is False:
            all_names.append([])
            #print(solution_steps)
    return(all_names)

def print_done_work(filename, solution_identifiers):
    if not os.path.isfile(filename):
        return
    print(bcolors.OKGREEN + "\nstart new drug-disease-pair" + bcolors.OKGREEN) 
    with open(filename, 'r') as jfile:
        data = json.load(jfile)
    for item in data['triples']:
        for triple in item:
            print(bcolors.OKBLUE + triple + bcolors.OKBLUE)
    print(bcolors.OKCYAN + str(solution_identifiers) + bcolors.OKCYAN)
    sys.exit(0)

def grade_grounding_step(expected, output_filename):
    """
    Create a plot showing
    """
    need_to_find = expected[1:3]
    need_to_find_id = [x.split(":")[1] for x in need_to_find]
    found = []
    print("\n", output_filename)
    #mesh_id, uniprot, mondo, hpo order for grounding storage
    with open(output_filename) as jfile:
        data = json.load(jfile)
        ground = data['grounded']
    for key, item in ground.items():
        for identifier in list(item):
            if identifier != "":
                identifier = identifier[0]
                if identifier in need_to_find_id:
                    index = need_to_find_id.index(identifier)
                    found.append(need_to_find[index])
                    #print(key, identifier)
    tp = len([x for x in need_to_find if x in found])
    fn = len([x for x in need_to_find if x not in found])
    print("true positives", tp)
    print("false negatives", fn)

def main():
    client = OpenAI()
    url = "https://en.wikipedia.org/w/api.php"
    node_norm_url = "https://nodenorm.transltr.io/get_normalized_nodes?curie="
    solution_filename = "data.tsv"
    solution_df = pd.read_table(solution_filename)
    num_abstracts = 20
    predicate_json = "./predicates/predicates.json"
    entity_json = "./predicates/entities.json"
    name_lookup_url = "https://name-lookup.transltr.io/lookup"
    indication_json = "./indication_paths.json"
    literature_dir = "./wiki_text"
    solution_names = parse_solutions(indication_json, solution_df)
    with open(entity_json, "r") as jfile:
        entity_data = json.load(jfile) 
    with open(predicate_json, "r") as jfile:
        predicate_data = json.load(jfile)

    for i, (index, row) in enumerate(solution_df.iterrows()):
        #print(i, "of", len(solution_df), "done.") 
        if i != 1:
            continue
        all_triples = []
        all_entities = []
        all_prompts = []
        solution_steps = solution_df.iloc[index].tolist()
        solution_identifiers = [x.lower() for x in solution_names[i]]
        if len(solution_identifiers) == 0:
            continue
        basename = solution_identifiers[0] + "_" + solution_identifiers[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)
        #if os.path.isfile(output_filename):
        #    grade_grounding_step(solution_steps, output_filename)
        #    continue
        #else:
        #    continue
        print("\n")
        print(bcolors.HEADER + str(solution_identifiers) + bcolors.HEADER)
        #first_check = alternate_path(solution_identifiers[0], solution_identifiers[-1], predicate_data)
        #sys.exit(0)

        with open("test_json.json", "r") as jfile:
            data = json.load(jfile)
        mechanism = data['mechansim']
        entities = data['entities']
        grounded = data['grounded']
        grounded_types = data['grounded_types']
        entity_list = entities.split(", ")
        entity_string = ""
        for entity in entity_list:
            entity_string += entity + "\n"
        print(entity_string)
        resp, prompt = prompts.triplets(mechanism, entity_string) 
        print(mechanism)
        print(bcolors.OKBLUE + resp)
        sys.exit(0)

        all_ranges = []
        range_dict = {}
        for key, value in predicate_data.items():
            if key not in range_dict:
                range_dict[key] = []
            if 'range' in value or "domain" in value:
                if 'range' in value:
                    all_ranges.append(value['range'])
                if 'domain' in value:
                    all_ranges.append(value['domain'])
                try:
                    range_dict[key] = [value['range'], value['domain']]
                except:
                    print(key, value)
            else:
                subclass_of = value['subclass']
                subclass_look = predicate_data[subclass_of]

                while "range" not in subclass_look and "domain" not in subclass_look:
                    subclass_of = subclass_look['subclass']
                    subclass_look = predicate_data[subclass_of]
                range_dict[key] = [subclass_look['range'], subclass_look['domain']]
        print(bcolors.OKBLUE + str(np.unique(all_ranges)))
        print(range_dict)
        #triples, prompt  = prompts.extract_predicates(paragraph, entity_string, predicate_string)
        sys.exit(0)
        with open(output_filename, 'w') as jfile:
            json.dump(first_check, jfile)
        #print_done_work(output_filename, solution_identifiers)
        #continue 

if __name__ == "__main__":
    main()
