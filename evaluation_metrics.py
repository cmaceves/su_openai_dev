"""
Evaluate random indications based on node groundings
"""
import os
import ast
import sys
import json
import random
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import util
import prompts
import wiki_pipeline
import no_wiki_pipeline
import color_scheme
from line_profiler import LineProfiler
random.seed(42)

type_dict = {"Biological Process" : "go", "Cell": "cell_ontology", "Cellular Component": "go", "Chemical Substance":"mesh", "Disease":"mesh", "Drug":"mesh", "Gene Family":"interpro", "Gross Anatomical Structure":"uberon", "Macromolecular Complex":"pr", "Molecular Activity":"go", "Organism Taxon": "ncbitaxon", "Pathway": "react", "Phenotypic Feature": "hp", "Protein": "uniprot", "Macromolecular Structure":"pr"} 
node_types = ['Biological Process', 'Cell', 'Cellular Component', 'Chemical Substance', 'Disease', 'Drug', 'Gene Family', "Gross Anatomical Structure", "Macromolecular Structure", "Molecular Activity", "Organism Taxon", "Pathway", "Phenotypic Feature", "Protein"]

def search_entity(search_value, database, n, real_vectors=None):
    """
    Given an entity and it's database, load the database and find the top n matches.
    """
    embedding_dir = '/home/caceves/su_openai_dev/embeddings'
    def_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
    #all_database_files = [os.path.join(embedding_dir, x) for x in os.listdir(embedding_dir) if "openai_%s"%database in x]
    #this version does not account for database
    all_database_files = [os.path.join(embedding_dir, x) for x in os.listdir(embedding_dir) if "ncbi" not in x]
    scores = []
    indexes = []
    files = []
    embedding_files = []
    #get the vector for the entity we're trying to ground
    embedding_1 = util.get_embeddings([search_value])
    embedding_1 = np.array(embedding_1)

    
    count = database.count("_")
    for db_file in all_database_files:
        loaded_array = util.load_array(db_file)
        distances = util.calculate_distances(loaded_array, embedding_1)
        index = range(0,len(distances))

        zipped = list(zip(distances, index))
        zipped.sort()

        distances, index = zip(*zipped)
        distances = list(distances)
        index = list(index)
        
        scores.extend(distances)
        indexes.extend(index)
        embedding_files.extend([db_file]*len(distances))

        db_file = "_".join(os.path.basename(db_file).split("_")[1:(3+count)])
            
        files.extend([db_file] * len(distances))

    zipped = list(zip(scores, indexes, files, embedding_files))
    zipped.sort()

    scores, indexes, files, embedding_files = zip(*zipped)
    scores = list(scores)
    indexes = list(indexes)
    files = list(files)
    embedding_files = list(embedding_files)

    return_names = []
    return_ids = []
    for i in range(n):
        def_file = os.path.join(def_dir, files[i] + ".jsonl")
        if not os.path.isfile(def_file):
            continue
        with open(def_file, 'r') as dfile:
            #for j, line in enumerate(dfile):
            #if j == indexes[i]:
            lines = dfile.readlines()
            line = lines[indexes[i]]
            data = json.loads(line)
            id_val = data['custom_id']
            name = data['body']['messages'][1]['content']
            return_names.append(name)
            return_ids.append(id_val)
            #print(scores[i], indexes[i], files[i], name, id_val)

    #let's compare to the aligned truth nodes
    comparison_scores = None
    if real_vectors is not None:
        tmp = util.load_array(embedding_files[0])[indexes[0]]
        tmp = tmp.reshape(1, -1)
        real_vectors = real_vectors.reshape(1, -1)
        dist = util.calculate_distances(real_vectors, tmp)
        comparison_scores = dist[0]

    return(return_names, return_ids, comparison_scores)

def search_embeddings(id_values):
    """
    Given a set of id values, find their vectors within our embedding space.
    """
    prefix_mapping = {"MESH":"mesh", "UniProt":"uniprot", "UBERON": "uberon", "GO": "go", "taxonomy": "ncbitaxon", "HP":"hp", "reactome":"reactome", "InterPro":"interpro"}
    batch_request_dir = "/home/caceves/su_openai_dev/batch_request_formatted"
    embedding_dir = "/home/caceves/su_openai_dev/embeddings"
    filenames = [""] * len(id_values)
    indexes = [0] * len(id_values)

    for i,id_val in enumerate(id_values):
        prefix = id_val.split(":")[0]
        suffix = id_val.split(":")[1]
        if prefix not in prefix_mapping:
            print("not mapped", id_val)
            continue
        mapped_db = prefix_mapping[prefix]
                
        #find the index and embedding fild of all mapped things
        all_possible_requests = [os.path.join(batch_request_dir, x) for x in os.listdir(batch_request_dir) if "%s_"%mapped_db in x]

        #search all possible files for the identifier we're looking for
        found = False
        if mapped_db == "uniprot":
            continue
        for request in all_possible_requests:
            with open(request, 'r') as rfile:
                for j,line in enumerate(rfile):
                    data = json.loads(line)
                    if ":" in data['custom_id']:
                        tmp = data['custom_id'].split(":")[1]
                    else:
                        tmp = data['custom_id']
                    if suffix == tmp:
                        indexes[i] = j
                        filenames[i] = request
                        found = True
                        break
            if found:
                break
        if not found:
            print(id_val, mapped_db)

    id_embeddings = {}
    #now we actually go to the embedding space and pull out the vector for the identifer
    for filename, index, id_val in zip(filenames, indexes, id_values):
        if filename == "":
            continue
        tmp = os.path.basename(filename).replace(".jsonl","")
        embedding_file = os.path.join(embedding_dir, "openai_%s_embeddings.npy"%tmp)
        loaded_array = util.load_array(embedding_file)
        id_embeddings[id_val] = loaded_array[index]
    return(id_embeddings)

def randomly_select_nodes(ground_truth_nodes, indications):
    """
    Pick negative control nodes for comparison.
    """
    all_nodes = []
    gt_ids = list(ground_truth_nodes.values())
    for indication in indications:
        nodes = indication['nodes']
        #node id values
        tmp = [x['id'] for x in indication['nodes']]
        for t, n in zip(tmp, nodes):
            if t not in gt_ids:
                all_nodes.append(n)
    random_selection = random.sample(all_nodes, len(gt_ids)) 
    return(random_selection)

def negative_control():
    """
    Here we compute cosine distance between randomly selected nodes from those already grounded.
    """
    alignment_dir = "./aligned_values"
    literature_dir = "./no_follow"
    indication_json = "indication_paths.json"
    figure_dir = "/home/caceves/su_openai_dev/su_figures"
    #evaluation_dir = "./evaluation_output"
    negative_dir = "/home/caceves/su_openai_dev/negative_controls"
    evaluation_indications = util.parse_indication_paths(indication_json)

    all_entries = [os.path.join(literature_dir, x) for x in os.listdir(literature_dir)]
    all_node_scores = []
    for i, entry in enumerate(all_entries):
        output_filename = os.path.join(negative_dir, os.path.basename(entry))
        if os.path.isfile(output_filename):
            continue
        output_dict = {}
        with open(entry, 'r') as efile:
            data = json.load(efile)
            
        #find the ground truth
        ground_truth_names = {}
        for key in evaluation_indications:
            drug  = key['graph']['drug'].lower().replace(" ", "_")
            disease  = key['graph']['disease'].lower().replace(" ","_")
            if drug in entry and disease in entry:
                ground_truth = [x['id'] for x in key['nodes']]
                for x in key['nodes']:
                    ground_truth_names[x['name']] = x['id']
                break

        comparison_nodes = randomly_select_nodes(ground_truth_names, evaluation_indications)
        comparison_nodes = [x['id'] for x in comparison_nodes]

        #this is the aligned pathway filename 
        drug_disease_string = drug.replace(" ","_") + "_" + disease.replace(" ","_")
        real_vectors = search_embeddings(ground_truth)
        fake_vectors = search_embeddings(comparison_nodes)

        rv_name = []
        fv_name = []
        scores = []
        for rv, fv in zip(real_vectors, fake_vectors):
            rv_tmp = real_vectors[rv]
            fv_tmp = fake_vectors[fv]
            rv_tmp = rv_tmp.reshape(1, -1)
            fv_tmp = fv_tmp.reshape(1, -1)
            distance = util.calculate_distances(rv_tmp, fv_tmp)
            rv_name.append(rv)
            fv_name.append(fv)
            scores.append(distance[0])
        print(output_filename)
        output_dict = {"ground_truth": rv_name, "randomized":fv_name, "scores":scores}
        with open(output_filename, 'w') as ofile:
            json.dump(output_dict, ofile)


def ground_nodes():
    """
    Here we attempt to ground nodes and then compare them to the ground truth version.
    """
    alignment_dir = "./aligned_values"
    literature_dir = "./no_follow"
    indication_json = "indication_paths.json"
    figure_dir = "/home/caceves/su_openai_dev/su_figures"
    evaluation_dir = "./evaluation_output"
    evaluation_indications = util.parse_indication_paths(indication_json)

    all_entries = [os.path.join(literature_dir, x) for x in os.listdir(literature_dir)]
    all_node_scores = []
    for i, entry in enumerate(all_entries):
        #TESTLINES
        if "clofedanol" not in entry:
            continue
        output_filename = os.path.join(evaluation_dir, os.path.basename(entry))
        #if os.path.isfile(output_filename):
        #    continue
        output_dict = {}
        #if i > 2:
        #    break
        with open(entry, 'r') as efile:
            data = json.load(efile)
        
        #find the ground truth
        ground_truth_names = {}
        for key in evaluation_indications:
            drug  = key['graph']['drug'].lower().replace(" ", "_")
            disease  = key['graph']['disease'].lower().replace(" ","_")
            if drug in entry and disease in entry:
                ground_truth = [x['id'] for x in key['nodes']]
                for x in key['nodes']:
                    ground_truth_names[x['name']] = x['id']
                break

        #this is the aligned pathway filename 
        drug_disease_string = drug.replace(" ","_") + "_" + disease.replace(" ","_")
        #TESTLINES this needs to be expanded to accomodate mulriple
        aligned_filenames = [x for x in os.listdir(alignment_dir) if drug_disease_string in x][0]
       
        #grounded column is the ground truth
        aligned_df = pd.read_table(os.path.join(alignment_dir, aligned_filenames))
        
        print(ground_truth_names)
        print(ground_truth)
        #dictionary of id to embedding vectors
        real_vectors = search_embeddings(ground_truth)

        """
        print(ground_truth)
        print(ground_truth_names)
        print(aligned_df)        
        print(real_vectors.keys())
        """
        num_aligned_entities = 0
        sum_comparison_scores = 0
        for entity, entity_type in data['entity_types'].items():
            database = type_dict[entity_type]            
            if database == "uniprot":
                continue
            entity_def = data['entity_definitions'][entity]
            search_value = entity + ": " + entity_def

            #comparison is to the aligned value
            aligned_entity = aligned_df[aligned_df['Entities'] == entity]['Grounded'].tolist()[0]
        
            if str(aligned_entity).lower() != "nan":
                #catalogging potential failure cases
                if aligned_entity not in ground_truth_names:
                    print("algined entity not in ground truth names")
                    continue
                aligned_id = ground_truth_names[aligned_entity]
                if aligned_id not in real_vectors:
                    print("not found in embedding", aligned_id)
                    continue
                gt_vector = real_vectors[aligned_id]
                num_aligned_entities += 1
            else:
                gt_vector = None
            #print("\n")
            #print("aligned entity", aligned_entity)
            print("entity", entity)
            return_names, return_ids, score = search_entity(search_value, database, 20, gt_vector)
            output_dict[entity] = {"return_names": return_names, "return_ids":return_ids, "score":score}
            print(return_names, return_ids, score)
            #sys.exit(0)
            if score is not None:
                sum_comparison_scores += score
            #print(score, return_names, return_ids)
        with open(output_filename, 'w') as ofile:
            json.dump(output_dict, ofile)
        print("sum score", sum_comparison_scores, "num aligned", num_aligned_entities)
        average_score = sum_comparison_scores / num_aligned_entities
        all_node_scores.append(average_score)

    sns.set_style("whitegrid")
    sns.histplot(x=all_node_scores, fill=True, color=color_scheme.PRIMARY_COLOR)
    plt.xlim(0,1)
    plt.ylabel("Distance from Ground Truth Node")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "embedding_scores_distribution.png"), dpi=300)

def main():
    name_lookup_url = "https://name-lookup.transltr.io/lookup"
    predicate_definition_file = "./predicates/predicate_definitions.json" #created by ChatGPT

    #randomly fetch n indications
    n = 50
    indication_json = "indication_paths.json"
    literature_dir = "./no_follow"

    with open(predicate_definition_file, 'r') as pfile:
        predicate_definitions = json.load(pfile)

    if not os.path.exists(indication_json):
        print("Downloading indications file...")
        indication_url = "https://raw.githubusercontent.com/SuLab/DrugMechDB/main/indication_paths.json"
        r = requests.get(indication_url, allow_redirects=True)
        open(indication_json, 'wb').write(r.content)
    
    evaluation_indications = util.parse_indication_paths(indication_json, n)
    
    for i, indication in enumerate(evaluation_indications):
        node_names = []
        node_ids = []
        for node in indication['nodes']:
            node_names.append(node['name'].lower())
        if node_names[0] != "clofedanol":
            continue
        basename = node_names[0] + "_" + node_names[-1] + ".json"
        basename = basename.replace(" ","_")
        output_filename = os.path.join(literature_dir, basename)     

        drug = indication['graph']['drug']
        disease = indication['graph']['disease']
        no_wiki_pipeline.main(drug, disease, indication, predicate_definition_file)
        sys.exit(0)
 
if __name__ == "__main__":
    main()
    #ground_nodes()
    #negative_control()
