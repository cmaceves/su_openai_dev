import os
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import color_scheme
import util


def plot_suggested_nodes():
    pass

def example_paths():
    """
    Given the dtw score, plot the best and worst pathways.
    """
    pass

def handle_id_inconsistencies(id_val):
    if id_val.startswith("D") and len(id_val) == 7:
        id_val = "MESH:" + id_val
    if "HP" in id_val:
        id_val = id_val.replace("HP_", "HP:")
    if id_val.startswith("P") and len(id_val) == 6:
        id_val = "UniProt:" + id_val
    return(id_val)

def confusion_matrix():
    input_dir = "./output2"
    indication_json = "indication_paths.json"
    figure_dir = "/home/caceves/su_openai_dev/su_figures"
    output_dir = ""
    all_indications = util.parse_indication_paths(indication_json)

    all_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    tp_counts = []
    fn_counts = []
    for filename in all_files:
        #output_filename = os.path.join(output_dir, os.path.basename(filename))
        #if os.path.isfile(output_filename):
        #    continue
        with open(filename, 'r') as rfile:
            data = json.load(rfile)

        output_values = []
        all_pathways = data['pathways']
        indication = data['indication']
        ground_truth_nodes = [x['name'] for x in indication['nodes']]
        ground_truth_ids = [x['id'] for x in indication['nodes']]

        grounding = data['grounding']

        print("\n")
        print("ground truth nodes", ground_truth_nodes)
        """
        print(data['paragraph'])
        print("entities", data['entities'])
        """
        print(ground_truth_ids)
        all_tp = []
        all_fn = []
        for pathway in all_pathways:
            found_gt = []
            tp = 0
            fn = 0
            graph_1_list = pathway.split(" -> ")
            graph_1_entities = graph_1_list[::2]
            print("\n")
            print(pathway)
            for entity in graph_1_entities:
                if entity not in grounding:
                    print(entity, "failed to ground")
                    continue
                best_id = grounding[entity]['ids'][0]
                mod_id = handle_id_inconsistencies(best_id)
                eq_ids = util.normalize_node(mod_id)
                #print("\n")
                #print(entity, "best id", best_id, "eq ids", eq_ids, "mod id", mod_id)
                #overlap with ground truth nodes
                gt = [x for x in eq_ids if x in ground_truth_ids if x != ground_truth_ids[0] and x != ground_truth_ids[-1]]
                if len(gt) > 0:
                    tp += 1
                    found_gt.append(gt[0])
            fn = len(ground_truth_ids) - 2 - len(found_gt)
            all_tp.append(tp/len(ground_truth_ids))
            all_fn.append(fn/len(ground_truth_ids))
        tp_counts.extend(all_tp)
        fn_counts.extend(all_fn)

        #print(all_tp)
        #print(all_fn)
        #sys.exit(0)
    print(tp_counts)
    samples = np.arange(len(tp_counts))
    sns.displot(tp_counts, color=color_scheme.PRIMARY_COLOR, fill=True)
    plt.xlabel("True Positive Percentage per Predicted Pathway")
    plt.savefig(os.path.join(figure_dir, "tp_fn_plot.png"), dpi=300)


if __name__ == "__main__":
    confusion_matrix()
