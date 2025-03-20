import os
import sys
import json
import calculate_embeddings

node_categorization_prompt = """You will be given a term and it's description. Please categorize it into one of the following categories. Return only the number of the assigned category with no additional commentary or formatting.
1. Biological Process
2. Cell
3. Cellular Component
4. Chemical Substance
5. Disease
6. Drug
7. Gene Family
8. Gross Anatomical Structure
9. Macromolecular Complex
10. Molecular Activity
11. Organism Taxon
12. Pathway
13. Phenotypic Feature
14. Protein"""
node_types = ['Biological Process', 'Cell', 'Cellular Component', 'Chemical Substance', 'Disease', 'Drug', 'Gene Family', "Gross Anatomical Structure", "Macromolecular Structure", "Molecular Activity", "Organism Taxon", "Pathway", "Phenotypic Feature", "Protein"]
def process_outputs(basename, output_dir, typed_dir):
    #parse the gpt outputs to dict format
    output_dict = {}
    filename = os.path.join(output_dir, basename + "_results.json")
    output_filename = os.path.join(typed_dir, basename + ".json")
    with open(filename, 'r') as rfile:
        for line in rfile:
            data = json.loads(line)
            id_val = data['custom_id']
            choice = data['response']['body']['choices'][0]['message']['content']
            try:
                choice = int(choice)-1
            except:
                continue
            node_type = node_types[choice]
            output_dict[id_val] = node_type

    with open(output_filename, 'w') as ofile:
        json.dump(output_dict, ofile)

def main():
    format_batch = True
    upload_batch = True
    check_batch = False
    process_typed_nodes = False
    database = "hp"
    database_path = "/home/caceves/su_openai_dev/parsed_databases/%s_definitions.json"%database
    format_dir = "/home/caceves/su_openai_dev/batch_request_inputs_categorize"
    output_dir = "/home/caceves/su_openai_dev/batch_request_outputs_categorize"
    typed_dir = "/home/caceves/su_openai_dev/typed_nodes"

    #format the batch
    if format_batch:
        calculate_embeddings.batch_request_formatting(database, database_path, format_dir, node_categorization_prompt, True)
    
    #upload the batch
    if upload_batch:
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database)]
        
        #upload the batches
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            calculate_embeddings.upload_batch(batch, basename, output_dir)

    if check_batch:
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database)]
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            calculate_embeddings.check_batch_status(basename, output_dir)
            calculate_embeddings.retrieve_batch(basename, output_dir)

    if process_typed_nodes:
        all_batches = [os.path.join(format_dir, x) for x in os.listdir(format_dir) if x.startswith(database)]
        for batch in all_batches:
            basename = os.path.basename(batch).replace(".jsonl", "")
            process_outputs(basename, output_dir, typed_dir)    
    
if __name__ == "__main__":
    main()
