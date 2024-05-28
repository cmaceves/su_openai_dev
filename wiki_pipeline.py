import re
import os
import sys
import ast
import copy
import nltk
import bioc
import json
import pickle
import requests
import pandas as pd
import numpy as np

import prompts
import pull_down_literature

from bs4 import BeautifulSoup
from itertools import permutations
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from openai import OpenAI
from line_profiler import LineProfiler
from wikidata.client import Client

lemmatizer = WordNetLemmatizer()

def wrapper_for_predicates(all_statements, predicate_list):
    kept_predicates = []
    response = Parallel(n_jobs=15)(delayed(prompts.describe_accuracy)(statement) for statement in all_statements) 
    for pred, resp in zip(predicate_list, response):
        if resp[0].lower() == "yes":
            kept_predicates.append(pred)
    return(kept_predicates)

def expand_entity_tree(entity_types, entity_data):
    """
    Given an entity category, traverse up the tree to get all possible classifications.
    """
    all_categories = []
    for entity in entity_types:
        all_categories.append(entity)
        if entity in entity_data:
            all_categories.extend(entity_data[entity])
    return(all_categories)


def camel_case_to_words(camel_case):
    #not currently in use, need better biolink predicate restrictions
    words = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', camel_case)
    new_word = ' '.join(words)
    return new_word.lower()

def find_predicates(predicate_data, entity_data, sub_types, obj_types):
    #not currently in use, need better biolink predicate restrictions
    predicate_list = []
    translated_sub = []
    for sub in sub_types:
        sub = sub.replace("biolink:","")
        new_sub = camel_case_to_words(sub)
        translated_sub.append(new_sub)
    translated_obj = []
    for obj in obj_types:
        obj = obj.replace("biolink:","")
        new_obj = camel_case_to_words(obj)
        translated_obj.append(new_obj)
    
    translated_obj = expand_entity_tree(translated_obj, entity_data)
    translated_sub = expand_entity_tree(translated_sub, entity_data)
    
    #domain is subject and range is object
    for key, value in predicate_data.items():
        useable = False
        if 'range' not in value and 'domain' not in value and 'subclass' not in value:
            continue
        range_val = ""
        domain_val = ""
        if 'range' in value:
            range_val = value['range']
        if 'domain' in value:
            domain_val = value['domain']

        if range_val == "" and "subclass" in value:
            subclass_of = value['subclass']
            subclass_look = predicate_data[subclass_of]
            while "range" not in subclass_look and "subclass" in subclass_look:
                subclass_of = subclass_look['subclass']
                subclass_look = predicate_data[subclass_of]
            if 'range' in subclass_look:
                range_val = subclass_look['range']

        if domain_val == "" and "subclass" in value:
            subclass_of = value['subclass']
            try:
                subclass_look = predicate_data[subclass_of]
            except:
                print(key, subclass_of)
                continue
            while "domain" not in subclass_look and "subclass" in subclass_look:
                subclass_of = subclass_look['subclass']
                subclass_look = predicate_data[subclass_of]
            if 'domain' in subclass_look:
                domain_val = subclass_look['domain']
         
        if len(translated_sub) > 0 and len(translated_obj) > 0:
            if domain_val in translated_obj and range_val in translated_sub:
                useable = True
            else:
                useable = False
        elif len(translated_sub) > 0 and len(translated_obj) == 0:
            if range_val in translated_sub:
                useable = True
            else:
                useable = False
        elif len(translated_obj) > 0 and len(translated_sub) == 0:
            if domain_val in translated_obj:
                useable = True
            else:
                useable = False
        if useable is True:
            predicate_list.append(key)
    return(predicate_list)

def find_standard_predicate(perm, grounded_types, predicate_restrictions, record_predicate_def, describe):
    sub = perm[0]
    obj = perm[1]
    sub_types = []
    obj_types = []
    if sub in grounded_types:
        sub_types = grounded_types[sub]
    if obj in grounded_types:
        obj_types = grounded_types[obj]
    sub_types = [x.replace("biolink:","") for x in np.unique(sub_types)]
    obj_types = [x.replace("biolink:","") for x in np.unique(obj_types)]

    predicate_list = []
    static_mappings = {"SmallMolecule":"Drug"}
    sub_types = [static_mappings[x] if x in static_mappings else x for x in sub_types]
    obj_types = [static_mappings[x] if x in static_mappings else x for x in obj_types]
    
    for key, value in predicate_restrictions.items():
        for val in value:
            if len(sub_types) > 1 and len(obj_types) > 1:
                if val[0] in sub_types and val[1] in obj_types:
                    if key not in predicate_list:
                        predicate_list.append(key)
            elif len(sub_types) > 1 and len(obj_types) == 0:
                if val[0] in sub_types:
                    predicate_list.append(key)
            elif len(obj_types) > 1 and len(sub_types) == 0:
                if val[1] in obj_types:
                    predicate_list.append(key)
            else:
                predicate_list.append(key)
    predicate_list = list(np.unique(predicate_list))
    statement_string = ""
    statement_numbers = []
    counter = 0

    #this limits the number of predicate with a first pass
    all_statements = []
    for i, predicate in enumerate(predicate_list):
        statement = sub + " " + predicate + " " + obj
        all_statements.append(statement)
    kept_predicates = wrapper_for_predicates(all_statements, predicate_list)

    #pick the best from the smaller list
    for i, predicate in enumerate(kept_predicates):
        statement = sub + " " + predicate + " " + obj
        if counter != 0:
            statement_string += "\n"
        statement_string += str(counter+1) + ". " + statement
        statement_numbers.append(predicate)
        counter += 1
        
    if len(statement_numbers) == 0:
        return(None)
    messages = [{"role":"system", "content":"You are a helpful assisant with biochemical and biomedical knowledge."}]
    for predicate in statement_numbers:
        messages.extend(record_predicate_def[predicate])
    #describe, prompt = prompts.describe_relationship(sub, obj)
    messages.append({"role": "user", "content": "How does '%s' relate to '%s'?"%(sub, obj)})
    messages.append({"role":"assistant", "content": "%s"%describe})
    resp, prompt = prompts.testy_test(statement_string, sub, obj, messages)
    if resp != 'no relationship': 
        try:
            found_statement = statement_numbers[int(resp)-1]
        except:
            return(None)
        #print(bcolors.OKGREEN + describe)
        #print(bcolors.OKGREEN + sub, bcolors.HEADER + statement_numbers[int(resp)-1], bcolors.OKGREEN + obj)
        return([sub, found_statement, obj])
    else:
        #print(bcolors.OKBLUE + "failure", bcolors.OKBLUE + sub, bcolors.OKBLUE + obj)
        return(None)

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
    for x in wiki_entity.iterlists():
        label = str(x[0].label)
        if label == "MeSH descriptor ID":
            mesh_id = x[1]
        if label == "UniProt protein ID":
            uniprot = x[1]
        if label == "Mondo ID":
            mondo = x[1]
        if label == "Human Phenotype Ontology ID":
            hpo = x[1]
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

def matched_wikipedia_pages(entity):
    useful_pages = [] 
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
        return({})
    try:
        ll = loose_data[1]
    except:
        print(entity)
        print("failure", loose_data)
        sys.exit(1)

    for possible_match in loose_data[1]:
        possible_match_compare = [lemmatizer.lemmatize(x, pos='n') for x in possible_match.lower().split(" ")]
        possible_match_compare = " ".join(possible_match_compare)
        entity_compare = [lemmatizer.lemmatize(x.lower(), pos='n') for x in entity.lower().split(" ")]
        entity_compare = " ".join(entity_compare)
        #print("here"+bcolors.OKBLUE, bcolors.OKGREEN+possible_match, bcolors.OKGREEN+entity, bcolors.HEADER + possible_match_compare, bcolors.HEADER + entity_compare)
        if possible_match_compare == entity_compare:
            useful_pages.append(possible_match)
    return({"entity":entity, "pages": useful_pages})


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


def get_mechanism(term1, term2):
    """
    Function to get the mechanism and entities associated.
    """
    print("fetching mechanism")
    wiki_client = Client() 
    #wikpedia query for the first term
    response, redirect, wikibase_id, linked_entities = query_wikipedia(term1)
 
    if redirect:
        response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
    tmp = response.split(".")
    paragraph1 = ""
    for sentence in tmp:
        paragraph1 += sentence.strip() + ". "

    #wikipedia quert for the second term
    response, redirect, wikibase_id, linked_entities = query_wikipedia(term2)
    if redirect:
        response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
    tmp = response.split(".")
    paragraph2 = ""
    for sentence in tmp:
        paragraph2 += sentence.strip() + ". "
    
    paragraph = paragraph1 + " " + paragraph2
    response, prompt = prompts.extract_mech_path(paragraph, term1, term2)
    response_list = response.split("\n")
    mechanism = response_list[0].replace("Mechanism: ", "")
    entities = response_list[-1].replace("Relevant Entities: ","").replace("Relevant Entities:","")
    entity_list = entities.split(", ")
    entity_list = [x.lower().replace(".","") for x in entity_list]
    return(mechanism, entity_list)

def alternate_path(term1, term2, predicate_data, filename):
    """
    1. Use both terms to query wikipedia, and derive a mechanism and important entities.
    2. Expand the entities to synonyms.
    3. Ground entities and their synonyms.
    """
    if os.path.isfile(filename):
        with open(filename, 'r') as jfile:
            output_data = json.load(jfile)
    else:
        output_data = {}
    
    #get mechanism and entities
    mech = True
    if mech:
        print("mech!")
        mechanism, entity_list = get_mechanism(term1, term2)
        output_data['mechanism'] = mechanism
        output_data['entities'] = entity_list
    else:
        mechanism = output_data['mechanism']
        entity_list = output_data['entities']

    #expand entities to synonyms
    syn = True
    if syn:
        print("syn!")
        synonym_dict = {}
        for entity in entity_list:
            resp, prompt = prompts.synonym_prompt(entity)
            synonym_list = resp.split("\n")
            synonym_list = [x.split(". ")[1].rstrip() for x in synonym_list]
            final_synonym_list = []
            #print(bcolors.HEADER+entity, bcolors.HEADER + str(synonym_list))
            for synonym in synonym_list:
                resp, prompt = prompts.synonym_context(entity, synonym, mechanism)
                if resp.lower() != "yes":
                    continue
                final_synonym_list.append(synonym)
                #print(bcolors.OKCYAN+resp, bcolors.OKBLUE+entity, bcolors.OKGREEN+synonym)
            synonym_dict[entity] = final_synonym_list
        output_data['synonyms'] = synonym_dict
    else:
        synonym_dict = output_data['synonyms']

    #find wikipedia pages for synonyms
    wiki = True
    if wiki:
        print("wiki!")
        synonym_pages_dict = {}
        pooled_synonyms = []
        for entity, synonym_list in synonym_dict.items():
            pooled_synonyms.extend(synonym_list)
            
        synonym_pages = Parallel(n_jobs=5)(delayed(matched_wikipedia_pages)(synonym) for synonym in pooled_synonyms) 
        tmp_dict = {}
        for synonym_search in synonym_pages:
            if len(synonym_search['pages']) > 0:
                tmp_dict[synonym_search['entity']] = synonym_search['pages']
        for entity, synonym_list in synonym_dict.items():
            for key, value in tmp_dict.items():
                if key in synonym_list:
                    if entity not in synonym_pages_dict:
                        synonym_pages_dict[entity] = []
                    synonym_pages_dict[entity].extend(value)
        output_data['synonym_pages'] = synonym_pages_dict
    else:
        synonym_pages_dict = output_data['synonym_pages']
    print("synonynm pages dict", synonym_pages_dict)
    #ground using wikipedia pages
    ground = True
    if ground:
        print("grounding!")
        synonym_groundings = {}
        for key, value in synonym_pages_dict.items():
            if key not in synonym_groundings:
                synonym_groundings[key] = {}
            for entity in value:
                if entity in synonym_groundings[key]:
                    continue
                print(entity)
                mesh_id, uniprot, mondo, hpo = ground_synonym(entity) 
                print(bcolors.OKBLUE + key, bcolors.OKCYAN + entity)
                print(bcolors.HEADER + str(mesh_id), uniprot, mondo, hpo)
                synonym_groundings[key][entity] = [mesh_id, uniprot, mondo, hpo]
        output_data['grounded'] = synonym_groundings
    else:
        synonym_groundings = output_data['grounded']

    #type the identifiers that we've grounded on
    type_ground = True
    if type_ground:      
        grounded_types = grounded_type(synonym_groundings) 
        output_data['grounded_type'] = grounded_types
    else:
        grounded_types = output_data['grounded_type']

    with open(filename, 'w') as jfile:
        json.dump(output_data, jfile)

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
    predicate_definitions = "./predicates/predicate_definitions.json" #created by ChatGPT
    entity_json = "./predicates/entities.json"
    name_lookup_url = "https://name-lookup.transltr.io/lookup"
    indication_json = "./indication_paths.json"
    prompt_save_dir = "./wiki_text/prompts_used"
    literature_dir = "./wiki_text"

    # Get indications file from DrugMechDB
    if not os.path.exists(indication_json):
        print("Downloading indications file...")
        indication_url = "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.json"
        r = requests.get(indication_url, allow_redirects=True)
        open(indication_json, 'wb').write(r.content)
    else:
        print("Found local indications file...")

    solution_names = parse_solutions(indication_json, solution_df)

    with open("indication_paths.json", "r") as jfile:
        data = json.load(jfile)

    drugmech_predicates = []
    drugmech_entities = []
    predicate_restrictions = {}
    for graph in data:
        for i,link in enumerate(graph['links']):
            sub = link['source']
            obj = link['target']
            predicate = link['key']
            for node in graph['nodes']:
                if node['id'] == sub:
                    sub_cat = node['label']
                if node['id'] == obj:
                    obj_cat = node['label']
            if predicate not in predicate_restrictions:
                predicate_restrictions[predicate] = []
            if [sub_cat, obj_cat] not in predicate_restrictions[predicate]:
                predicate_restrictions[predicate].append([sub_cat, obj_cat])
            drugmech_entities.append(sub_cat)
            drugmech_entities.append(obj_cat)
            drugmech_predicates.append(predicate)
    drug_mech_entities = list(np.unique(drugmech_entities))
    drug_mech_predicates = list(np.unique(drugmech_predicates))

    with open(entity_json, "r") as jfile:
        entity_data = json.load(jfile) 
    with open(predicate_json, "r") as jfile:
        predicate_data = json.load(jfile)

    #here we have ChatGPT define all our predicates, and then dump the responses to a json file
    need_def_predicates = False
    if need_def_predicates:
        record_predicate_def = {}
        #define all the possible predicates
        for i, predicate in enumerate(list(predicate_restrictions.keys())):
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            template = {"role": "user", "content": "What does '%s' mean for entities in a biochemical or biomedical context?"%predicate}
            messages.append(template)
            resp, prompt = prompts.grammatical_check(messages)
            other_template = {"role":"assistant", "content":"%s"%resp}
            messages.append(other_template)
            record_predicate_def[predicate] = messages[1:]

        with open(predicate_definitions, 'w') as pfile:
            json.dump(record_predicate_def, pfile) 
    else:
        with open(predicate_definitions, 'r') as pfile:
            record_predicate_def = json.load(pfile)
    
    if not os.path.exists(literature_dir):
        os.makedirs(literature_dir)
        print(f"Directory '{literature_dir}' was created.")
    else:
        print(f"Directory '{literature_dir}' already exists.")
            
    for i, (index, row) in enumerate(solution_df.iterrows()):
        print("\n")
        print(i, "of", len(solution_df), "done.") 
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

        if not os.path.isfile(output_filename):
            #grade_grounding_step(solution_steps, output_filename)
            continue
            
        print(bcolors.HEADER + str(solution_identifiers) + bcolors.HEADER)

        #first_check = alternate_path(solution_identifiers[0], solution_identifiers[-1], predicate_data, output_filename)
        #continue
        #sys.exit(0)       
        
        with open(output_filename, "r") as jfile:
            data = json.load(jfile)
        mechanism = data['mechanism']
        entities = data['entities']
        grounded = data['grounded']
        grounded_types = data['grounded_type']

        print(bcolors.OKCYAN + mechanism)
        print(bcolors.OKBLUE + str(entities))
        #given a set of entities, lets explore the relationship between each of them
        perm = permutations(entities, 2)
        unique_permutations = set(perm)
        triplets = [] 
        for j,perm in enumerate(unique_permutations):
            print(j, "of", len(unique_permutations))
            triple = find_standard_predicate(perm, grounded_types, predicate_restrictions, record_predicate_def, mechanism)
            if triple is not None:
                triplets.append(triple)
                print(bcolors.OKCYAN + str(triple))

        data['triplets'] = triplets
        with open(output_filename, "w") as jfile:
            json.dump(data, jfile)

if __name__ == "__main__":
    main()
