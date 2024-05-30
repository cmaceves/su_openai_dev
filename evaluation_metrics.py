"""
Evaluate random indications based on node grodunings and predicates.
"""
import os
import sys
import json
import random
random.seed(42)

def parse_indication_paths(indication_json, n=10):
    """
    indication_json : string
        The full path to the file containing the known indiciation paths.
    n : int 
        The number of indiciations to randomly choose for evaluation.
    """
    with open(indication_json, 'r') as jfile:
        data = json.load(jfile)

    selected_items = random.choices(data, k=n)    
    print(len(selected_items)) 

def main():
    #develop a system for storing the evaluation including prompts in a dated way

    #model type

    #randomly fetch n indications
    n = 10
    indication_json = "indication_paths.json"

    if not os.path.exists(indication_json):
        print("Downloading indications file...")
        indication_url = "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.json"
        r = requests.get(indication_url, allow_redirects=True)
        open(indication_json, 'wb').write(r.content)
    parse_indication_paths(indication_json, n)

    #evaluate based on node groundings 
    
    #evaluate based on predicate similarity


if __name__ == "__main__":
    main()
