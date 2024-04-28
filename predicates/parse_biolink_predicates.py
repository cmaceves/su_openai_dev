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

entities = {}
for key, value in data['classes'].items():
    if key not in entities:
        entities[key] = []
    if 'is_a' in value:
        parent = value['is_a']
        entities[key].append(parent)
        while parent in data['classes']:
            tmp = data['classes'][parent]
            if 'is_a' in tmp:
                parent = tmp['is_a']
                entities[key].append(parent)
            else:
                parent = ""
with open(entity_definition_file, 'w') as jfile:
    json.dump(entities, jfile)

predicate_dict = {}
useful_is_a = []
for key, value in data['slots'].items():
    if key == "node property":
        print(key, value)
    if 'is_a' in value and value['is_a'] in useful_is_a:
        #print(key, "-", value['is_a'])
        continue
    if key == "synonym":
        print(key, value)
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

