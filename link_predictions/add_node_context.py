import os
import sys
import pandas as pd
import upload_batches

def main():
    node_file = "nodes.csv"
    node_df = pd.read_csv(node_file)
    entities = node_df['name'].tolist()
    system_prompt = """You will be given a biological or chemical entity. Please list all contexts where the entity could occur at the organ level, the tissue level, the cellular level, and the sub-cellular compartment level. Multiple answers may be given in each context category, be exhaustive. Do not include information in parentheses. If context cannot be assigned for a category, return "None".  Return in json format, as follows:{
  organ: [],
  tissue: [],
  cell_type: [],
  sub_cellular_component: []
 }"""
    output_prefix = "context_queries"
    params_prefix = "context_params"
    results_prefix = "context_output"
    #upload_batches.batch_request_formatting(entities, output_prefix, system_prompt)

    all_batches = os.listdir(output_prefix)
    upload = True
    check = False
    if upload:
        for batch in all_batches:
            batch_id = batch.replace(".jsonl","")
            upload_batches.upload_batch(os.path.join(output_prefix, batch), batch_id, params_prefix)
    if check:
        for batch in all_batches:
            batch_id = batch.replace(".jsonl","")
            check_batch_status(batch_id, params_prefix)
            retrieve_batch(basename, params_prefix, results_prefix)

if __name__ == "__main__":
    main()
