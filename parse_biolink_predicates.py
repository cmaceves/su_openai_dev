import yaml
import json

with open("biolink-model.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

predicates = []
for key, value in data['slots'].items():
    if 'annotations' in value:
        if 'canonical_predicate' in value['annotations']:
            if value['annotations']['canonical_predicate']:
                predicates.append(key)
predicate_dict = {}
predicate_dict['predicates'] = predicates
with open("predicates.json", "w") as jfile:
    json.dump(predicate_dict, jfile)
