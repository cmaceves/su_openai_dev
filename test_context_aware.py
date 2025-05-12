"""
Here we test a few examples to see if we can capture context for mechanisms contained within DrugMechDB.
"""
import os
import sys
import json

import prompts
import util

def parse_out_pairs(evaluation_indications, entity_type_one, entity_type_two):
    #get pairs of entity-entity interactions and their disease context
    pairs_return = []
    for indication in evaluation_indications:
        links = indication['links']
        nodes = indication['nodes']
        disease = indication['graph']['disease']
        for link in links:
            sub = link['source']
            obj = link['target']
            
            sub_type = [x for x in nodes if x['id'] == sub][0]['label'].lower()
            obj_type = [x for x in nodes if x['id'] == obj][0]['label'].lower()

            if sub_type == entity_type_one and obj_type == entity_type_two:
                sub_name = [x for x in nodes if x['id'] == sub][0]['name'].lower()
                obj_name = [x for x in nodes if x['id'] == obj][0]['name'].lower()
                tmp = [sub_name, obj_name, disease.lower()]
                pairs_return.append(tmp)

    return(pairs_return)

def main():
    n = 50
    indication_json = "indication_paths.json"
    entity_type_one = "protein"
    entity_type_two = "protein"

    evaluation_indications = util.parse_indication_paths(indication_json, n)
    pairs_return = parse_out_pairs(evaluation_indications, entity_type_one, entity_type_two)

    for i, pair in enumerate(pairs_return):
        if i > 5:
            break
        resp, prompt = prompts.pair_context(pair[0], pair[1], pair[2])
        print(pair)
        print(resp)

if __name__ == "__main__":
    main()
