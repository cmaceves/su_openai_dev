import os
import calculate_embeddings

node_categorization_prompt = """You will be given a term. Please categorize it into one of the following categories. Return only the number of the assigned category with no additional commentary or formatting.
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

def main():
    format_batch = False
    upload_batch = True
    check_batch = False
    database = "mesh"
    database_path = "/home/caceves/su_openai_dev/parsed_databases/%s_definitions.json"%database
    format_dir = "/home/caceves/su_openai_dev/batch_request_inputs_categorize"
    output_dir = "/home/caceves/su_openai_dev/batch_request_outputs_categorize"
    #format the batch
    if format_batch:
        calculate_embeddings.batch_request_formatting(database, database_path, format_dir, node_categorization_prompt)
    
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

if __name__ == "__main__":
    main()
