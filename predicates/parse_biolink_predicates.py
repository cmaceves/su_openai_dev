import ast
import sys
import yaml
import json
import nltk
import pandas as pd
import numpy as np

with open("biolink-model.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
entity_definition_file = "./entities.json"

"""
useful_definitions = ["anatomical entity", "anatomical entity to anatomical entity association", "biological entity", "biological process", "biological process or activity", "chemical substance", "disease or phenotypic feature", "gene or gene product", "gene product", "genomic entity","molecular entity", "protein"]
all_definitions = []
entities = {}
for key, value in data['classes'].items():
    if 'is_a' in value:
        all_definitions.append(value['is_a'])
        if value['is_a'] in useful_definitions:
            if 'description' in value:
                entities[key] = value['description']
                print(key, value)
#print(np.unique(all_definitions))
with open(entity_definition_file, 'w') as jfile:
    json.dump(entities, jfile)
sys.exit(0)
"""
predicate_dict = {}
useful_is_a = ['node property', 'association slot', 'aggregate statistic', 'interbase coordinate']
for key, value in data['slots'].items():

    if ('range' not in value or 'domain' not in value) and "is_a" not in value:
        print(key, value)
        continue
    if 'is_a' in value and value['is_a'] in useful_is_a:
        #print(key, "-", value['is_a'])
        continue
    if key not in predicate_dict:
        predicate_dict[key] = {}
    if "description" in value:
        predicate_dict[key]["definition"] = value['description']
    if "is_a" in value:
        predicate_dict[key]["subclass"] = value['is_a']
    if "domain" in value:
        predicate_dict[key]['domain'] = value['domain']
    if 'range' in value:
        predicate_dict[key]["range"] = value['range']
#print(predicate_dict.keys())
with open("predicates.json", "w") as jfile:
    json.dump(predicate_dict, jfile)

