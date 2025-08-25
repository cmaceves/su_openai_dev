"""
Put all databases in the same format.
"""
import re
import os
import csv
import sys
import json
import libsbml #parse sbml file format
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
go_db = "/home/caceves/su_openai_dev/databases/go_definitions.txt"
output_dir = "/home/caceves/su_openai_dev/parsed_databases"

record = []
all_records = []

def process_record(record):
    keep = True
    output_dictionary = {}
    function = []
    passed_end = False
    print_status=False
    found_id=False
    for line in record:
        #if print_status:
        #    print(line)
        if line.startswith("DE   "):
            if print_status:
                print(line)
            tmp = line.split("   ")[1]
            tmp = tmp.split(";")
            tmp = [x for x in tmp if "RecName" in x]
            if len(tmp) == 0:
                continue
            if len(tmp) > 1:
                print(tmp)
            if "RecName: Full=" in tmp[0]:
                tmp = tmp[0].split("RecName: Full=")[1]
                if print_status:
                    print(tmp)
                output_dictionary['name'] = tmp
        if line.startswith("AC   "):
            if found_id:
                continue
            tmp = line.split("   ")[1].split(";")[0]
            if print_status:
                print(tmp)
            output_dictionary["accession"] = tmp
            found_id = True
        if line.startswith("CC"):
            if passed_end:
                continue
            tmp = line.split("   ")[1:]
            if "---------------------------------------------------------------------------" in tmp:
                passed_end = True
                continue
            if "CC   -!-" in line and "FUNCTION" not in line:
                passed_end = True
                continue
            #if print_status:
            #    print(line)
            function.extend(tmp)
        if line.startswith("OS   "):
            tmp = line.split("   ")
            species = tmp[1][:-1]
    function = (" ").join(function).replace("-!- FUNCTION:","")
    function = re.sub(r"\{.*?\}", "", function)
    function = re.sub(r"\(.*?\)", "", function)
    function = function.replace(" .","").replace("  ", " ")
    output_dictionary['function'] = "Belongs to %s species."%(species) + function
    return(keep, output_dictionary)

def process_uniprot():
    all_records = []
    record = []
    seen_accessions = []
    uniprot_db = "/home/caceves/su_openai_dev/databases/uniprot_sprot.dat"
    with open(uniprot_db, "r") as dfile:
        for i, line in enumerate(dfile):
            line = line.strip()
            if line == "//":
                if len(record) > 0:
                    keep, output_dictionary = process_record(record)
                    if output_dictionary['accession'] == "P35348":
                        print(output_dictionary)
                    #if output_dictionary['accession'] in seen_accessions:
                    #    keep = False
                    #    continue
                    if keep:
                        all_records.append(output_dictionary)
                        seen_accessions.append(output_dictionary['accession'])
                record = []
            else:
                record.append(line)
    print(len(all_records))
    with open(os.path.join(output_dir, "uniprot_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)

def process_go():
    #parse name, accession, function
    all_records = []
    with open(go_db, 'r') as gfile:
        for line in gfile:
            output_dict = {}
            line = line.strip()
            line_list = line.split("\t")
            output_dict['name'] = line_list[1]
            output_dict['accession'] = line_list[0]
            output_dict['function'] = line_list[2]
            all_records.append(output_dict)

    with open(os.path.join(output_dir, "go_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)

def process_hp():
    hp_db = "/home/caceves/su_openai_dev/databases/hp.json"

    with open(hp_db, "r") as hfile:
        data = json.load(hfile)
    all_records = []
    graphs = data['graphs']
    for value in graphs:
        print(len(value['nodes']))
        for i, node in enumerate(value['nodes']):
            if i == 0:
                continue
            if 'type' not in node or node['type'] != "CLASS":
                continue
            output_dict = {}
            id_val = node['id']
            id_val = os.path.basename(id_val)
            if 'lbl' not in node:
                print(node)
                sys.exit(0)
            label = node['lbl']
            if "meta" in node and "definition" in node['meta']:
                description = node['meta']['definition']['val']
            else:
                description = ""
            output_dict['accession'] = id_val
            output_dict['name'] = label
            output_dict['function'] = description
            all_records.append(output_dict)

    with open(os.path.join(output_dir, "hp_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)


def process_mesh():
    mesh_db = "/home/caceves/su_openai_dev/databases/desc2025.xml"
    # Load XML file
    tree = ET.parse(mesh_db)
    root = tree.getroot()

    all_records = []
    for record in root.findall("DescriptorRecord"):
        output_dict = {}
        mesh_id = record.find("DescriptorUI").text
        mesh_name = record.find("DescriptorName/String").text
        scope_note = record.find(".//ConceptList/Concept/ScopeNote")
        description = scope_note.text if scope_note is not None else ""
        description = description.strip()
        output_dict['accession'] = mesh_id
        output_dict['name'] = mesh_name
        output_dict['description'] = description
        all_records.append(output_dict)

    with open(os.path.join(output_dir, "mesh_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)

def process_uberon():
    uberon_db = "/home/caceves/su_openai_dev/databases/uberon.owl"
    tree = ET.parse(uberon_db)
    root = tree.getroot()
    all_records = []

    ns = {'owl': 'http://www.w3.org/2002/07/owl#', 'rdfs': 'http://www.w3.org/2000/01/rdf-schema#', 'obo': 'http://purl.obolibrary.org/obo/'}

    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)


    for term in root.findall(".//{http://www.w3.org/2002/07/owl#}Class"):
        term_id = term.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")

        if term_id and "UBERON_" in term_id:
            uberon_id = "UBERON:" + term_id.split("_")[-1]  # Convert URI to UBERON:XXXXXX format

            label_elem = term.find("{http://www.w3.org/2000/01/rdf-schema#}label")
            name = label_elem.text if label_elem is not None else ""

            definition_elem = term.find("{http://purl.obolibrary.org/obo/}IAO_0000115")
            description = definition_elem.text if definition_elem is not None else ""
            all_records.append({"accession": uberon_id, "name": name, "description": description})

    with open(os.path.join(output_dir, "uberon_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)

def process_ncbitaxon():
    ncbi_db = "/home/caceves/su_openai_dev/databases/names.dmp"
    nodes_db = "/home/caceves/su_openai_dev/databases/nodes.dmp"

    all_records = {}
    with open(ncbi_db, "r", encoding="utf-8") as filename:
        reader = csv.reader(filename, delimiter="|")
        for row in reader:
            output_dict = {}
            tax_id = row[0].strip()  # Taxon ID
            name = row[1].strip()  # Scientific or common name
            name_type = row[3].strip()  # Type of name

            if tax_id not in all_records:
                output_dict = {"accession": tax_id, "name": "", "description": ""}
                all_records[tax_id] = output_dict

            if name_type == "scientific name":
                all_records[tax_id]['name'] = name
            elif name_type == "common name":
                all_records[tax_id]['description'] = name
            elif name_type == "genbank common name":
                all_records[tax_id]['description'] = name

    with open(nodes_db, "r", encoding="utf-8") as filename:
        reader = csv.reader(filename, delimiter="|")
        for row in reader:
            tax_id = row[0].strip()
            blast_name = row[8].strip()
            if tax_id in all_records:
                if blast_name:
                    if len(blast_name) > 1:
                        all_records[tax_id]["description"] = blast_name if blast_name else ""

    final_all_records = []
    for key, item in all_records.items():
        final_all_records.append(item)

    with open(os.path.join(output_dir, "ncbitaxon_definitions.json"), "w") as jfile:
        json.dump(final_all_records, jfile)

def process_cell_ontology():
    cell_db = "/home/caceves/su_openai_dev/databases/cell_ontology.owl"
    tree = ET.parse(cell_db)
    root = tree.getroot()

    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'obo': 'http://purl.obolibrary.org/obo/',
        'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#'
    }

    all_records = []

    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    for term in root.findall(".//{http://www.w3.org/2002/07/owl#}Class"):
        term_id = term.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")

        if term_id and "CL_" in term_id:
            accession = "CL:" + term_id.split("_")[-1]  # Convert URI to UBERON:XXXXXX format

            label_elem = term.find("{http://www.w3.org/2000/01/rdf-schema#}label")
            name = label_elem.text if label_elem is not None else ""

            definition_elem = term.find("{http://purl.obolibrary.org/obo/}IAO_0000115")
            description = definition_elem.text if definition_elem is not None else ""
            #print(name, description)

            output_dict = {"accession": accession, "name": name, "description": description}
            #print(output_dict)
            all_records.append(output_dict)

    with open(os.path.join(output_dir, "cell_ontology_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)

def process_reactome():
    reactome_dir = "/home/caceves/su_openai_dev/databases/reactome"
    all_pathway_files = [os.path.join(reactome_dir, x) for x in os.listdir(reactome_dir)]
    all_records = []
    for filename in all_pathway_files:
        document = libsbml.readSBML(filename)
        model = document.getModel()

        id_val = os.path.basename(filename).replace(".sbml","")
        name = model.getName()

        notes_html = model.getNotesString()
        soup = BeautifulSoup(notes_html, "xml")
        paragraph_text = soup.get_text().strip()

        output_dict = {}
        output_dict['accession'] = id_val
        output_dict['name'] = name
        output_dict['description'] = paragraph_text

        all_records.append(output_dict)

    with open(os.path.join(output_dir, "reactome_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)


def process_pr():
    pr_database = "/home/caceves/su_openai_dev/databases/protein_ontology.owl"
    tree = ET.parse(pr_database)
    root = tree.getroot()
    ns = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'obo': 'http://purl.obolibrary.org/obo/',
        'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#'}

    all_records = []

    # Extract all rdf:Description elements, which represent ontology terms
    for term in root.findall(".//{http://www.w3.org/2002/07/owl#}Class"):
        term_id = term.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")

        if term_id and "PR_" in term_id:  # Ensure it's a Protein Ontology term
            accession = "PR:" + term_id.split("_")[-1]  # Convert URI to UBERON:XXXXXX format

            label_elem = term.find("{http://www.w3.org/2000/01/rdf-schema#}label")
            name = label_elem.text if label_elem is not None else ""

            definition_elem = term.find("{http://purl.obolibrary.org/obo/}IAO_0000115")
            description = definition_elem.text if definition_elem is not None else ""

            #print(accession, name, description)
            output_dict = {"accession": accession, "name": name, "description": description}
            print(output_dict)
            all_records.append(output_dict)

    with open(os.path.join(output_dir, "pr_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)

def process_interpro():
    filename = "/home/caceves/su_openai_dev/databases/interpro.xml"
    tree = ET.parse(filename)
    root = tree.getroot()
    all_records = []

    for entry in root.findall("interpro"):
        accession = entry.get("id")

        name = entry.find("name").text if entry.find("name") is not None else ""

        abstract = entry.find("abstract/p")
        if abstract is not None:
            description = "".join(abstract.itertext()).strip()
            description = re.sub(r"\[[^\[\]]*\]", "", description).strip()
        else:
            description = ""

        output_dict = {}
        output_dict['accession'] = accession
        output_dict['name'] = name
        output_dict['description'] = description

        all_records.append(output_dict)
    with open(os.path.join(output_dir, "interpro_definitions.json"), "w") as jfile:
        json.dump(all_records, jfile)


def process_chebi():
    filename = "/home/caceves/su_openai_dev/databases/names_3star.tsv"
    df = pd.read_csv(filename, sep="\t", dtype=str)
    all_records = []

    for index, row in df.iterrows():
        output_dict = {}
        accession = row['ID']
        name = row['NAME']

        output_dict['accession'] = accession
        output_dict['name'] = name
        output_dict['description'] = ""

        all_records.append(output_dict)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chebi_definitions.json")
    with open(output_file, "w") as jfile:
        json.dump(all_records, jfile)

def stats_master_db():
    filepath = "/home/caceves/su_openai_dev/parsed_databases"
    json_files = [os.path.join(filepath, x) for x in os.listdir(filepath)]

    for filename in json_files:
        with open(filename, 'r') as jfile:
            data = json.load(jfile)
        print(filename, len(data))

if __name__ == "__main__":
    #process_uniprot()
    #process_go()

    #process_hp()
    #process_mesh()
    #process_uberon()
    #process_ncbitaxon()
    #process_cell_ontology()
    #process_reactome()
    #process_pr()
    #process_interpro()
    process_chebi()
    #stats_master_db()

