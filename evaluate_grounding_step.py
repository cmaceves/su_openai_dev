import os
import sys
import json
import util
import numpy as np
import pandas as pd
import predict_identfier_embedding
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

def align_lists(cosine_similarity_matrix):
    # Get the dimensions of the cosine similarity matrix
    num_items_list1, num_items_list2 = cosine_similarity_matrix.shape
    
    # Calculate the number of items to be removed
    if num_items_list1 > num_items_list2:
        items_to_remove_from_list1 = num_items_list1 - num_items_list2
        items_to_remove_from_list2 = 0
    else:
        items_to_remove_from_list1 = 0
        items_to_remove_from_list2 = num_items_list2 - num_items_list1
    
    # Use linear sum assignment (Hungarian algorithm) to find the optimal alignment
    # This will give us the best matching pairs based on the cosine similarity matrix
    row_ind, col_ind = linear_sum_assignment(-cosine_similarity_matrix)
    
    # Create sets of all indices for both lists
    all_indices_list1 = set(range(num_items_list1))
    all_indices_list2 = set(range(num_items_list2))
    
    # The matched items are the ones we want to keep
    matched_items_list1 = set(row_ind)
    matched_items_list2 = set(col_ind)
    
    # Items to be removed are those not matched
    removed_from_list1 = list(all_indices_list1 - matched_items_list1)[:items_to_remove_from_list1]
    removed_from_list2 = list(all_indices_list2 - matched_items_list2)[:items_to_remove_from_list2]
    
    return removed_from_list1, removed_from_list2

def basic_confusion(eval_identifiers, grounded_identifiers):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for eid in eval_identifiers:
        if eid in grounded_identifiers:
            true_positive += 1
        else:
            false_negative += 1

    for gid in grounded_identifiers:
        if gid not in eval_identifiers:
            false_positive += 1
    return(true_positive, false_positive, false_negative)

def graph_alignment(nodes_1, nodes_2):

    vectors_1 = util.try_openai_embeddings(nodes_1)
    vectors_2 = util.try_openai_embeddings(nodes_2)

    cs = cosine_similarity(vectors_1, vectors_2)
    remove_1, remove_2 = align_lists(cs)
    
    score_nodes_1 = [x for i,x in enumerate(nodes_1) if i not in remove_1]
    score_nodes_2 = [x for i,x in enumerate(nodes_2) if i not in remove_2]
    return(score_nodes_1, score_nodes_2)

def main():
    directory_name = "/home/caceves/su_openai_dev/no_wiki_text"
    all_json_files = [os.path.join(directory_name, x) for x in os.listdir(directory_name) if 'prompts' not in x]

    not_grounded_nodes = {"node_names":[]}
    indication_json = "indication_paths.json"
    evaluation_indications = util.parse_indication_paths(indication_json)

    #TESTLINES
    no_length = 0
    length = 0

    #evaluate all nodes for the test set
    matches = 0
    mismatches = 0
    not_attempted = 0
    for i, json_file in enumerate(all_json_files):
        with open(json_file, 'r') as jfile:
            data = json.load(jfile)

        grounded_nodes = data['grounded_nodes']
        triples = data['triples']
        
        all_nodes = list(grounded_nodes.keys())
        drug = all_nodes[0]
        disease = all_nodes[-1]
        print(drug, disease)
        #print(triples)
        #find the correct answer
        found = False
        for eval_net in evaluation_indications:
            tmp_drug = eval_net['graph']['drug']
            tmp_disease = eval_net['graph']['disease']
            if tmp_drug.lower() == drug.lower() and tmp_disease.lower() == disease.lower():
                found = True
                eval_nodes = eval_net['nodes']
                break
        if not found:
            continue

        if len(eval_nodes) != len(grounded_nodes):
            no_length += 1
        else:
            length += 1

        grounded_identifiers = list(grounded_nodes.values())
        eval_identifiers = [x['id'] for x in eval_nodes]
        all_eval_nodes = [x['name'] for x in eval_nodes]    

        true_positive, false_positive, false_negative = basic_confusion(eval_identifiers, grounded_identifiers)
        score_nodes_eval, score_nodes_exp = graph_alignment(all_eval_nodes, all_nodes)
           
        #score the aligned graphs
        eval_dict = dict(zip(all_eval_nodes, eval_identifiers))               
        print("\neval", score_nodes_eval)
        print("exp", score_nodes_exp)
        for gt, exp in zip(score_nodes_eval, score_nodes_exp):
            exp_id = grounded_nodes[exp]
            gt_id = eval_dict[gt]
            if exp_id != gt_id and exp_id != "":
                print(gt_id, exp_id)
                print(gt, exp)
                mismatches += 1
            if exp_id == "":
                not_attempted += 1
            if exp_id == gt_id:
                matches += 1
        #matches = matches-2
        #sys.exit(0)
        #search_literature_support()

    print("match", matches, "mis", mismatches, "not tried", not_attempted)

if __name__ == "__main__":
    main()



