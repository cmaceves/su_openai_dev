import re
import os
import sys
import ast
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

categories = ["protien", "receptor", "enzyme", "cell", "disease", "gene family", "endogenous small molecule"]

def extract_triplicates(predicate_json, abstract, triplet_file):
    if os.path.isfile(triplet_file):
        return
    #once we have entities categorized, we want to extract the predicate relationships from the text
    
    with open(predicate_json, 'r') as jfile:
        data = json.load(jfile)
    with open(abstract, "r") as afile:
        abstracts = json.load(afile)['abstracts']

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
        if pmid != "9264044":
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
            sentence = prompts.sentence_tense(sentence, pos)
            triplets = prompts.triplicates(sentence)
            triplets = triplets.split('\n')
            print(sentence)
            print(triplets)
            all_triplets.extend(triplets)
        triplet_dict[pmid] = all_triplets
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
        for j in range(1):
            entities = prompts.entity_query(initial_text)
            try:
                entities = ast.literal_eval(entities)
            except:
                continue
            entities = [x.lower() for x in entities]
            contexts = []
            for entity in entities:
                resp = prompts.entity_context(initial_text, entity)
                contexts.append(resp)
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

class subj:
    def __init__(self):
        self.name = ""
        self.biologically_relevant = ""
        self.unique_identifier = ""
        self.is_original = False
    def set_biological_relevance(self, bio):
        self.biologically_relevant = bio

class obj:
    def __init__(self):
        self.name = ""
        self.biologically_relevant = ""
        self.unique_identifier = ""
        self.is_original = False

    def set_biological_relevance(self, bio):
        self.biologically_relevant = bio
        

class pred:
    def __init__(self):
        self.name = ""
        self.biologically_relevant = "" #TODO change this to be more specific
        self.unique_identifier = ""

    def set_biological_relevance(self, bio):
        self.biologically_relevant = bio

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

def main():
    client = OpenAI()
    solution_filename = "data.tsv"
    solution_df = pd.read_table(solution_filename)
    num_abstracts = 20
    predicate_json = "./predicates/predicates.json"
    name_lookup_url = "https://name-lookup.transltr.io/lookup"
    indication_json = "./indication_paths.json"
    solution_names = parse_solutions(indication_json, solution_df)

    for i, (index, row) in enumerate(solution_df.iterrows()):
        if index > 5:
            break

        solution_steps = solution_df.iloc[index].tolist()
        """
        solution_identifiers = []
        for sol in solution_steps:
            if "UniProt" in sol:
                sol = sol.replace("UniProt", "UniProtKB")
            all_identifiers, x = get_normalizer([sol])    
            tmp = all_identifiers[0].lower()
            solution_identifiers.append(tmp)
        print(solution_steps)
        print(solution_identifiers)
        continue
        """
        solution_identifiers = [x.lower() for x in solution_names[i]]
        #happens with longers paths
        if len(solution_identifiers) == 0:
            continue
        triplet_objects = []
        literature_dir = "./text_data" #storage location of pubmed docs

        #make dir if doesn't exist
        if not os.path.isdir(literature_dir):
            os.system("mkdir {data_dir}".format(data_dir=literature_dir))

        """
        all_text = ""
        #alternative path for using GPT to summarize text
        if os.path.isfile(abstract):
            with open(abstract, 'r') as rfile:
                data = json.load(rfile)
            if 'abstracts' not in data:
                continue
            for i, (key, initial_text) in enumerate(data['abstracts'].items()):
                all_text += initial_text.strip()
        print(len(all_text.split(" ")))
        #print(all_text)
        resp = prompts.summarize_text(all_text)
        print(resp)
        print(len(resp.split(" ")))
        sys.exit(0)
        """
        #resp = requests.post(name_lookup_url, params={"string":"human histamine h1 receptors"})
        #print(resp.json())
        term_string = solution_identifiers[0] + " " + solution_identifiers[-1] + " mechanism"
        term_list = [solution_identifiers[0], solution_identifiers[-1]]
        expansion(term_string, term_list, literature_dir, num_abstracts, predicate_json)
        sys.exit(0)
    
    """
    for original_identifier in term_list:
        all_identifiers, _ = get_normalizer([original_identifier])
        all_identifiers = list(np.unique([x.lower() for x in all_identifiers]))
        original_content = all_identifiers[0]     
        
    resp = requests.post(name_lookup_url, params={"string":original_content}) 
    identifiers = resp.json()
    all_matches = []
    match = identifiers[0]
    all_matches.append(match['curie'])
    all_matches.append(match['label'])
    all_matches.extend(match['synonyms'])
    all_matches = [x.lower() for x in all_matches]
    
    print(original_content)
    """
if __name__ == "__main__":
    main()
