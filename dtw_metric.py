import os
import sys
import json
import copy
import util
import random
import prompts
import numpy as np
from tslearn.metrics import dtw, dtw_path
from line_profiler import LineProfiler

def select_random_node(ground_truth_ids, all_indications, n=1):
    filtered_indications = []
    for indication in all_indications:
        nodes = indication['nodes']
        node_ids = [x['id'] for x in nodes]
        keep_nodes = [x for x,y in zip(nodes, node_ids) if y not in ground_truth_ids]
        filtered_indications.extend(keep_nodes)
    add_nodes = random.sample(filtered_indications, n)
    return(add_nodes)


def main():
    input_dir = "./output2"
    indication_json = "indication_paths.json"
    output_dir = "dtw_metrics"
    all_indications = util.parse_indication_paths(indication_json)

    index_path='/home/caceves/su_openai_dev/index_map.pkl'
    index_map = util.load_index_map(index_path)
    total_rows = sum(count for _, count in index_map)
    reverse_map = []
    for file, count in index_map:
        reverse_map.extend([(file, i) for i in range(count)])
    first_file = index_map[0][0]
    arr = np.load(first_file, mmap_mode='r')
    embedding_dim = arr.shape[1]

    all_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    for filename in all_files:
        output_filename = os.path.join(output_dir, os.path.basename(filename))
        if os.path.isfile(output_filename):
            continue
        with open(filename, 'r') as rfile:
            data = json.load(rfile)

        output_values = []
        all_pathways = data['pathways']
        indication = data['indication']
        ground_truth_nodes = [x['name'] for x in indication['nodes']]
        ground_truth_ids = [x['id'] for x in indication['nodes']]
        ground_truth_vectors = []
        add_nodes = select_random_node(ground_truth_ids, all_indications, 3)
        for node, id in zip(ground_truth_nodes, ground_truth_ids):
            resp, prompt = prompts.define_entity(node, "gpt-4.1")
            return_names, return_ids, comparison_scores, top_embedding = util.search_entity(resp, 10, reverse_map=reverse_map, embedding_dim=embedding_dim, total_rows=total_rows)
            ground_truth_vectors.append(top_embedding)
        print("got gt")
        #for our random nodes, fetch the ground truth vector
        random_node_vectors = []
        for node in add_nodes:
            resp, prompt = prompts.define_entity(node['name'], "gpt-4.1")
            return_names, return_ids, comparison_scores, top_embedding = util.search_entity(resp, 10, reverse_map=reverse_map, embedding_dim=embedding_dim, total_rows=total_rows)
            random_node_vectors.append(top_embedding)
        print("got add node")
        for pathway in all_pathways:
            tmp = {}
            print(pathway)
            predicted_vectors = [ground_truth_vectors[0]]
            graph_1_list = pathway.split(" -> ")
            graph_1_entities = graph_1_list[::2]
            if len(graph_1_entities) <= 3:
                continue
            for node in graph_1_entities:
                if node in data['grounding']:
                    top_embedding = data['grounding'][node]['embedding']
                    predicted_vectors.append(top_embedding)
            predicted_vectors.append(ground_truth_vectors[-1])
            if len(predicted_vectors) != len(graph_1_entities):
                print("not same length")
                continue
            P = np.array(predicted_vectors)
            G = np.array(ground_truth_vectors)

            position = random.randint(1, len(predicted_vectors) - 2)
            SP1_path = copy.deepcopy(graph_1_entities)
            SP1_path.insert(position, add_nodes[0]['name'])
            SP1 = copy.deepcopy(predicted_vectors)
            SP1.insert(position, random_node_vectors[0])
            SP1 = np.array(SP1)

            SP3 = copy.deepcopy(predicted_vectors)
            SP3_path = copy.deepcopy(graph_1_entities)

            if len(predicted_vectors) <= 4:
                positions = random.choices(range(1, len(predicted_vectors) - 1), k=3)
            else:
                positions = random.sample(range(1, len(predicted_vectors) - 1), 3)
            positions.sort(reverse=True)

            for i,(pos, item) in enumerate(zip(positions, random_node_vectors)):
                SP3.insert(pos, item)
                SP3_path.insert(pos, add_nodes[i]['name'])
            SP3 = np.array(SP3)
            optimal_path, dtw_score = dtw_path(P, G)
            print("optimal path", optimal_path)
            print("dtw score", dtw_score)

            optimal_path, dtw_score_sp1 = dtw_path(SP1, G)
            print("sp1 optimal path", optimal_path)
            print("sp1 dtw score", dtw_score_sp1)

            optimal_path, dtw_score_sp3 = dtw_path(SP3, G)
            print("sp3 optimal path", optimal_path)
            print("sp3 dtw score", dtw_score_sp3)
            tmp['score'] = dtw_score
            tmp['score_sp1'] = dtw_score_sp1
            tmp['score_sp3'] = dtw_score_sp3

            tmp['predicted_path'] = pathway
            tmp['sp1_path'] = SP1_path
            tmp['sp3_path'] = SP3_path

            output_values.append(tmp)

        with open(output_filename, 'w') as wfile:
            json.dump(output_values, wfile)


if __name__ == "__main__":
    main()
