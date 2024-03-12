import re
import os
import sys
import ast
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
categories = ["protien", "receptor", "enzyme", "cell", "disease", "gene family", "endogenous small molecule"]

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
            linked_entities.append(title.lower())
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

def extract_triplicates(predicate_json, abstract, triplet_file):
    #if os.path.isfile(triplet_file):
    #    return
    #once we have entities categorized, we want to extract the predicate relationships from the text
    
    with open(predicate_json, 'r') as jfile:
        data = json.load(jfile)
    with open(abstract, "r") as afile:
        abstracts = json.load(afile)['abstracts']
    if len(abstracts) == 0:
        return
    #get possible parts of speech
    pos_tag_df = pd.read_table("pos_tag.tsv",sep=";")
    parts_of_speech = {}
    for index, row in pos_tag_df.iterrows():
        init_def = row['Meaning']
        init_def = re.sub("[\(\[].*?[\)\]]", "", init_def)
        if "," in init_def:
            init_def = init_def.split(",")[0]
        if init_def[-1] == " ":
            init_def = init_def[:-1]
        parts_of_speech[row['Abbreviation']] = init_def

    #extract the triplicates
    triplet_dict = {}
    for i, (pmid, text_body) in enumerate(abstracts.items()):  
        print(pmid)
        text_body = prompts.text_tense(text_body)
        text_body = prompts.shorten_sentences(text_body)
        text_body = prompts.replace_pronouns(text_body)
        print(text_body)
        triplets = prompts.triplets(text_body)
        print(triplets.lower())
        continue
        sentences = text_body.split(".")    
        all_triplets = []
        for i, sentence in enumerate(sentences):
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            pos = ""
            for word in pos_tags:
                if word[1] in parts_of_speech:
                    pos += parts_of_speech[word[1]] + ";"
            #sentence = prompts.sentence_tense(sentence, pos)

            triplets = prompts.triplicates(sentence)
            triplets = triplets.lower()
            print(triplets)
            triplets = triplets.split('\n')
            #print(sentence)
            #print(triplets)
            
            all_triplets.extend(triplets)
        triplet_dict[pmid] = all_triplets
    sys.exit(0)
    with open(triplet_file, "w") as jfile:
        json.dump(triplet_dict, jfile)

def categorize_entities(ce, all_entities):
    if os.path.isfile(ce):
        return
    print("categorizing entities")
    lemmatizer = WordNetLemmatizer()
    questions_by_pmid = {}
    final_entities = {}
    final_categories = {}
    for i, (pmid, entities) in enumerate(all_entities.items()):
        if pmid not in final_entities:
            final_entities[pmid] = []
            final_categories[pmid] = []
            questions_by_pmid[pmid] = []
        entities = [lemmatizer.lemmatize(x) for x in entities]

        for j, entity in enumerate(entities):
            for k, cat in enumerate(categories):
                tmp = "Is %s a %s?" %(entity, cat)
                questions_by_pmid[pmid].append(tmp)

    for i, (pmid, questions) in enumerate(questions_by_pmid.items()):
        print(i, "of", len(questions_by_pmid))
        question_string = ""
        token_count = 0
        queries = [] #this is the question list
        responses = [] #this stores the corresponding responses
        counter = 1
        for j, question in enumerate(questions):                
            if token_count > 2500:
                question_string = question_string[:-1]
                results = prompts.entity_categorization(question_string)
                results_list = results.split("\n")
                #print(question_string)
                question_string = ""
                token_count = 0
                counter = 1
                try:
                    results_list = [x.split(".")[1] for x in results_list]
                except:
                    print("failed", results_list)
                    continue
                responses.extend(results_list)
            queries.append(question)
            question_string += "%s."%str(counter) + question + "\n"
            counter += 1
            token_count += len(question_string.split(" ")) + 1

        #clean up and lignering queries
        question_string = question_string[:-1]
        results = prompts.entity_categorization(question_string)
        results_list = results.split("\n")
        #print(question_string)
        question_string = ""
        token_count = 0
        results_list = [x.split(".")[1] for x in results_list]
        responses.extend(results_list)
        if len(responses) != len(queries):
            print(len(responses), len(queries))
            print("error")
            continue

        #now let's figure out which entities we keep in final_entites and final_categories
        for resp, quer in zip(responses, queries):
            if resp == "yes":
                ent = quer.split(" a ")[0].replace("Is ","")
                cat = quer.split(" a ")[1].replace("?","")
                #print(ent, cat)
                final_entities[pmid].append(ent)
                final_categories[pmid].append(cat)
    
    with open(ce, 'w') as cfile:
        json.dump({"categorized_entities":final_entities, "categories":final_categories}, cfile)

def pull_entities(es, abstract):
    if os.path.isfile(es):
        return
    print("extracting entities")
    all_entities = {}
    all_contexts = {}
    with open(abstract, 'r') as rfile:
        data = json.load(rfile)
    if 'abstracts' not in data:
        return
    for i, (key, initial_text) in enumerate(data['abstracts'].items()):
        print("pulling entities %s of %s" %(str(i), str(len(data['abstracts']))))
        for j in range(1):
            entities = prompts.entity_query(initial_text)
            try:
                entities = ast.literal_eval(entities)
            except:
                continue
            entities = [x.lower() for x in entities]
            contexts = []
            #for entity in entities:
            #    resp = prompts.entity_context(initial_text, entity)
            #    contexts.append(resp)
            all_entities[key] = entities
            all_contexts[key] = contexts
    with open(es, "w") as jfile:
        json.dump({"entities":all_entities, "contexts": all_contexts}, jfile)

def expansion(term_string, term_list, literature_dir, num_abstracts, predicate_json):
    file_ext = "{term}.json".format(term=term_string.replace(" ","_").replace("/","").replace("(", "").replace(")",""))

    pull_down_literature.fetch_pubmed_abstracts(term_string, literature_dir, num_abstracts, term_list)
    #open these text files and do entity extraction
    abstract = os.path.join(literature_dir, file_ext)
    es = os.path.join(literature_dir, "entity_storage_{term}".format(term=file_ext))
    pull_entities(es, abstract)
    with open(es, "r") as efile:
        data = json.load(efile)
        all_entities = data['entities']
    ce = os.path.join(literature_dir, "categorized_entities_%s"%file_ext)
    categorize_entities(ce, all_entities)
    te = os.path.join(literature_dir, "triplets_%s"%file_ext)
    extract_triplicates(predicate_json, abstract, te)

def parse_biolink_definitions(filename="biolink_def.txt"):
    resp = {}
    with open(filename, 'r') as tfile:
        for line in tfile:
            line = line.strip()
            line_list = line.split("\t")
            if len(line_list) < 2:
                continue
            if line_list[1] == "None":
                continue
            sc = snake_case_split(line_list[0])
            sc = " ".join(sc)
            tmp = sc.lower() + " - " + line_list[1].lower() + "; "
            tmp.lower()
            resp[sc] = tmp

    return(resp)

class Entity:
    def __init__(self):
        self.name = ""
        self.mesh_id = []
        self.category = ""

class subj:
    def __init__(self):
        self.name = ""
        self.mesh_id = ""

class obj:
    def __init__(self):
        self.name = ""
        self.mesh_id = ""       

class pred:
    def __init__(self):
        self.name = ""

class triplet:
    def __init__(self):
        self.subj = subj()
        self.pred = pred()
        self.obj = obj()
        self.useful = True

    def set_triplet(self, triplet):
        if len(triplet) != 3:
            return
        self.subj.name = triplet[0]
        self.pred.name = triplet[1]
        self.obj.name = triplet[2]

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
    print("\n")
    with open(filename, 'r') as jfile:
        data = json.load(jfile)
    for key, item in data['entities'].items():
        print(bcolors.OKBLUE + key + bcolors.OKBLUE)
        print(bcolors.OKGREEN + str(item) + bcolors.OKGREEN)

    print(bcolors.OKCYAN + str(solution_identifiers) + bcolors.OKCYAN)
    #sys.exit(0)

def main():
    wiki_client = Client() 
    client = OpenAI()
    url = "https://en.wikipedia.org/w/api.php"

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
        data = json.load(jfile)
    predicate_string = ""
    predicate_list = []
    for key, value in data.items():
        predicate = key.lower()
        predicate_list.append(predicate)
        #print(predicate, value)
        if 'definition' not in value:
            #print(key, value['subclass'])
            #continue
            defintion = predicate
        else:
            definition = value['definition'].lower()
        predicate_string += predicate + "\n"  #":" + definition + "\n"
        #predicate_string += predicate + "\n"
    #print(predicate_string)
    #sys.exit(0)
    
    for i, (index, row) in enumerate(solution_df.iterrows()):
        #print(i, "of", len(solution_df), "done.")
        """
        if index < 3:
            continue
        if index > 3:
            break
        """
        solution_steps = solution_df.iloc[index].tolist()
        solution_identifiers = [x.lower() for x in solution_names[i]]
        #happens with longers paths
        if len(solution_identifiers) == 0:
            continue
        basename = solution_identifiers[0] + "_" + solution_identifiers[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)
        #print_done_work(output_filename, solution_identifiers)
        #continue

        response, redirect, wikibase_id, linked_entities = query_wikipedia(solution_identifiers[0])
        entity = Entity()
        entity.name = solution_identifiers[0]
        if redirect:
            response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
        all_triples = {}
        all_entities = {}
        all_prompts = []
        if wikibase_id != "":
            wiki_entity = wiki_client.get(wikibase_id, load=True)
            for x in wiki_entity.keys():
                child = wiki_client.get(x.id)
                label = str(child.label).strip()
                if label == "MeSH descriptor ID":
                    entity.mesh_id = wiki_entity.getlist(child)
                #child_list = wiki_entity.getlist(child)
                """
                for child_entity in child_list:
                    #if type(child_entity) == "wikidata.entity.Entity":                    
                    print(child.label)
                    print(type(child_entity))
                    print(child_list)
                """
        triples, prompt  = prompts.extract_predicates(response, linked_entities, predicate_string)
        print(response)
        print(triples.lower())
        triples = triples.lower().split("\n")
        print(linked_entities)
        sys.exit(0)
        response, redirect, wikibase_id, linked_entities = query_wikipedia(solution_identifiers[-1])
        if redirect:
            response, redirect, wikibase_id, linked_entities = query_wikipedia(response)
        
        if wikibase_id != "":
            wiki_entity = wiki_client.get(wikibase_id, load=True)
            for x in wiki_entity.keys():
                child = wiki_client.get(x.id)
                label = str(child.label).strip()
                if label == "MeSH descriptor ID":
                    entity.mesh_id = wiki_entity.getlist(child)
                child_list = wiki_entity.getlist(child)
                print(label)
                print(child_list)
        triples, prompt  = prompts.extract_predicates(response, predicate_string)
        triples = triples.lower().split("\n")
        print(response)
        print(triples)
        sys.exit(0)
        """
        sys.exit(0)
        resp, prompt = prompts.test(response, solution_identifiers[0], solution_identifiers[-1])
        all_prompts.append(prompt)
        entities, prompt = prompts.entity_query(resp)
        all_prompts.append(prompt)
        try:
            entity_list = ast.literal_eval(entities)
        except:
            continue
        entity_list = [x.lower() for x in entity_list]
        all_entities[solution_identifiers[0]] = entity_list
        triples, prompt = prompts.triplets(resp, entities)
        triples = triples.split("\n")
        all_prompts.append(prompt)
        triples = [x.lower() for x in triples]
        all_triples[solution_identifiers[0]] = triples
        """
        """
        resp, prompt = prompts.test(response, solution_identifiers[0], solution_identifiers[-1])
        all_prompts.append(prompt)
        entities, prompt = prompts.entity_query(resp)
        all_prompts.append(prompt)
        try:
            entity_list = ast.literal_eval(entities)
        except:
            continue
        entity_list = [x.lower() for x in entity_list]
        all_entities[solution_identifiers[-1]] = entity_list
        triples, prompt = prompts.triplets(resp, entities)
        triples = triples.split("\n")
        all_prompts.append(prompt)
        triples = [x.lower() for x in triples]
        all_triples[solution_identifiers[-1]] = triples

        with open(output_filename, "w") as jfile:
            json.dump({'entities': all_entities, 'triples':all_triples, "prompts":all_prompts}, jfile)

        sys.exit(0)
        """
        """
        for j, entity in enumerate(entity_list):
            print("%s of %s done." %(str(j), str(len(entity_list))))
            if entity in seen_wiki_search:
                continue
            seen_wiki_search.append(entity)
            text_data = query_wikipedia(entity)
            tmp_all_triples = []
            batches = []
            count = 0
            text_tmp = ""
            tmp = text_data.split("\n")
            for paragraph in tmp:
                if count > 2000:
                    batches.append(text_tmp)
                    text_tmp = ""
                    count = 0
                text_tmp += " " + paragraph.strip()
                count += len(paragraph.split(" ")) 
            for batch in batches:
                text_body = prompts.text_tense(batch)
                text_body = prompts.shorten_sentences(text_body)
                text_body = prompts.replace_pronouns(text_body)
                entities = prompts.entity_query(text_body)
                try:
                    entities = [x.lower() for x in ast.literal_eval(entities)]
                except:
                    continue
                triples = prompts.triplets(text_body, entities)
                triples = [x.lower() for x in triples.split("\n")]
                tmp_all_triples.extend(triples)
            all_triples[entity] = tmp_all_triples
        with open(output_filename, "w") as jfile:
            json.dump(all_triples, jfile)
        """

if __name__ == "__main__":
    main()
