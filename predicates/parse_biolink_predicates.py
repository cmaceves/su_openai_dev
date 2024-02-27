import ast
import sys
import yaml
import json
import nltk
import pandas as pd

with open("biolink-model.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
predicate_dict = {}
for key, value in data['slots'].items():
    if key not in predicate_dict:
        predicate_dict[key] = {}
    if "description" in value:
        predicate_dict[key]["definition"] = value['description']
    if "is_a" in value:
        predicate_dict[key]["subclass"] = value['is_a']
    if "domain" in value:
        predicate_dict[key]['domain'] = value['domain']
with open("predicates.json", "w") as jfile:
    json.dump(predicate_dict, jfile)

