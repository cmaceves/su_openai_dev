import os
import sys
import ast
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def pair_context(entity1, entity2, disease, drug):
    prompt =[
          {"role": "system", "content": 
              """You will be given two biological or chemical entities that interact in the mechanism of action of a drug, and the disease state being treated by this interaction. Please provide additional context about these two proteins in terms of what subcellular components they co-localize in, what cell types they co-localize in, what tissue types they co-localize in, and what organs type they co-localize in. Please create a json object where the keys are 'subcellular components', 'cell types', 'tissue types', and 'organs'. If these two entities do no co-localize of no information can be found about a co-localize in a category, please set the value of that category to 'None'. If these two entities do not interact within this disease state, please set all values to 'None'."""},
          {"role": "user", "content": "Entity 1:%s\nEntity 2:%s\nEntity 3:%s\nDrug: %s"%(entity1, entity2, disease, drug)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt,
    temperature = 0,
    response_format={"type": "json_object" }
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)



def fuzzy_nodes(entity_string):
    #given a set of biological entities, "fuzzy" them up.
    prompt =[
          {"role": "system", "content": 
              """You will be given a numbered list of biological and chemical entities. Your job is to "fuzzy" each entity by changing the word or phrase while still maintaining the same meaning. Please create a json object where the key is the input entity and the value is the "fuzzy" version of that entity."""},
          {"role": "user", "content": "%s"%(entity_string)}
      ]

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    temperature = 0,
    response_format={"type": "json_object" }
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def define_nodes(node):
    prompt =[
          {"role": "system", "content": 
           """Given a biochemical entity describe it and define it's function in under 150 tokens"""},
          {"role": "user", "content": "%s"%(node)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt,
    temperature = 0,
    max_tokens=200
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def type_nodes(node):
    prompt =[
          {"role": "system", "content": 
           """
           You will be given a term and it's description. Please categorize it into one of the following categories. Return only the number of the assigned category with no additional commentary or formatting.
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
14. Protein
           """},
          {"role": "user", "content": "%s"%(node)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def chain_of_thought_5(paragraph, predicate_string, target_entity, entity_string):
    prompt =[
          {"role": "system", "content": 
           """
Task:
You will be given a paragraph, a target entity, a numbered list of controlled predicates, and a numbered list of biological entities. Your job is to extract semantic triples from the paragraph related to the target entity.
- Semantic triples consist of two entities and a predicate describing the relationship between them.
- The first entity of all extracted semantic triples must be the target entity.
- The second entity must be a biological or chemical entity and can be proteins, receptors, small molecules, organisms, drugs, diseases, phentotypic features, macromolecular complexes, cell types, molecular activities, biological processes, gene families, or pathways and will be provided in list form.
- Predicates must be selected from the provided list.
- Second entities must be selected from the provided list.
- Never return more than one semantic triple where the second entity is identical. For example if the triple 'Entity -> Predicate1 -> OtherEntity1' is returned then do NOT return 'Entity -> Predicate2 -> OtherEntity1' because both of these have a second entity of 'OtherEntity1' even if the predicate term is different.
- Do not return the same semantic triple twice.
- Please return the semantic triples as a numbered list with no additional formatting or commentary.
- If no semantic triples can be extracted please return "None"

Example Input Format:
Text
Target Entity: Entity
Predicates:
1. Predicate1
2. Predicate2
3. Predicate3
Entities:
1. OtherEntity1
2. OtherEntity2
3. OtherEntity3
4. OtherEntity4

Example Output Format:
1. Entity -> Predicate3 -> OtherEntity1
2. Entity -> Predicate1 -> OtherEntity2
3. Entity -> Predicate3 -> OtherEntity3
           """},
          {"role": "user", "content": "%s\nTarget Entity: %s\nPredicates:\n%s\nEntities:\n%s"%(paragraph, target_entity, predicate_string, entity_string)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def get_pathway(drug, diease, mechanism, links):
    messages = [{"role": "system", "content": 
    """You will be given a drug, a disease, a list of links, and a paragraph describing the mechanism of action of how a drug treats the disease. Links are directional semantic triples that consist of two biological entities, separated by a predicate which describes the relationship between them. Links can be strung together to form longer paths, in this case the second entity of the first link would need to be the first entity of the second link. For example the links:
1. Entity1 - > Predicate1 -> Entity2
2. Entity2 -> Predicate2 -> Entity3
Could be strung together to create the path:
Entity1 -> Predicate1 -> Entity2 -> Predicate2 -> Entity3.
Your task is to take the given list of links, and find all paths that lead from the drug to the disease and describe the mechanism of action of the drug treating the disease. Paths must begin with the drug and end with the disease, and no entity may be repeated multiple times within a path. Use only the links and entities provided in the "Links" section. Return all paths both direct and indirect including paths with up to 15 links All paths must be unique. Return each path on a new line and no additional formatting or characters. Please return up to fifteen unique paths.
    """},
    {"role":"user", "content":"Drug:%s\nDisease:%s\nMechanism:%s\nLinks:%s"%(drug, diease, mechanism, links)}]

    completion = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
    response = str(completion.choices[0].message.content)
    return(response)


def define_database_term(term):
    messages = [{"role": "system", "content": 
    """Given a biochemical entity describe it and define it's function in under 150 tokens"""},
    {"role":"user", "content":"%s"%(term)}]

    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0, max_tokens=200)
    response = str(completion.choices[0].message.content)
    return(response)

def find_correct_identifier(node, synonym_string):
    messages = [{"role": "system", "content": 
    """you will be given an entity, and ten possible synonyms for that entity. please pick the closest match, return only that term exactly with no additional commentary. each synonym is provided on a new line, and a returned match must identically and fully match one of the provided synonyms. do not shorten, abbrieviate, or change the returned synonym. if no synonym is found, return "none"."""},
    {"role":"user", "content":"entity:%s\nsynonyms:\n%s"%(node, synonym_string)}]

    completion = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0, max_tokens=200)
    response = str(completion.choices[0].message.content)
    return(response)

def find_correct_identifier_openai(node, synonym_string):
    messages = [{"role": "system", "content": 
    """You will be given a biological entity and its description, and a list of synonym descriptions that might also match the entity. Each synonym and its description is provided on a new line. Please return only the number of the closest synonym description with no additional commentary. If not match can be found, return "None"."""},
    {"role":"user", "content":"entity description:%s\nsynonym descriptions:\n%s"%(node, synonym_string)}]

    completion = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0, max_tokens=200)
    response = str(completion.choices[0].message.content)
    return(response)


def relationships(question):
    messages = [{"role": "system", "content": """
    Given a question about two biological entities, please answer it in paragraph form. If no answer can be found please return "No Answer".
    """},
    {"role":"user", "content":
    """
    %s
    """%(question)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)

def chain_of_thought_4(drug, disease):
    prompt =[
          {"role": "system", "content": 
           """
           Given a drug and the disease it treats please describe in paragraph form the mechanism of action. Use simple sentences and be as descriptive as possible. Do not return any additional formatting or commentary.
           """},
          {"role": "user", "content": "Drug: %s\nDisease: %s\n"%(drug, disease)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def chain_of_thought_answer_questions(questions):
    messages = [{"role": "system", "content": """
    Please answer the following questions. Return your answers as a numbered list with no additional commentary or formatting.
    """},
    {"role":"user", "content":
    """
    %s
    """%(questions)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)

def determine_translator_synonym(target, labels):
    messages = [{"role": "system", "content": """
    You will be given a target term, and a numbered list of terms that potentially mean the same thing. Please determine which term in the list is most similar to the target term. Please return the number of the closest matching term only, with no additional commentary or formatting. Do not return the term itself.
    """},
    {"role":"user", "content":
    """
    Target Term: %s
    Potential Matches: %s
    """%(target, labels)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def chain_of_thought_questions(drug, disease, graph):
    messages = [{"role": "system", "content": """
    Task:
You will be given a drug, the disease it treats, and a paragraph describing how the drug's mechanism of action.  A mechanism of action is the specific biological process through which a drug, chemical, or other substance produces its effects on a cell, tissue, organ, or organism. It describes how the substance interacts with molecular targets, such as receptors, enzymes, ion channels, or nucleic acids, to initiate a cascade of events leading to its physiological or therapeutic outcome. Please generate a list of questions, which if answered would help add additional context and information to the drug's specific mechanism of action. Tailor this list of questions to get more intermediate steps to the mechanism focused around specific biochemical entities. Return the questions as a numbered list with no additional commentary.
    """},
    {"role":"user", "content":
    """
    Drug: %s
    Disease: %s
    Paragraph: 
    %s
    """%(drug, disease, graph)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def extract_triples(paragraph, entity_1, predicates):
    messages = [{"role": "system", "content": """
    Task:
    You will be given a paragraph of text, a target entity, and a list of predicates. Your task is to extract semantic triples which contain the target entity from the text.

    - A semantic triple is made up of two biological or chemical entities, connected by a predicate.
    - Each semantic triple should describe a relationship that can stand independently.
    - Both entities in a semantic triple must be biological or chemical entities such as proteins, receptors, macromolecular complexes, diseases, drugs, small molecules, organisms, phenotypic features, pathways, biological processes, genes, cellular components, anatomical structures, or cell types.
    - The predicate describes the relationship between the entities.
    - Only use the predicates provided as input to describe the relationships.
    - Triples should be returned seperately and formatted with an entity, the predicate, and then a second entity in the following format: Entity -> Predicate -> Entity
    - Wxpand abbreviations to full entity names.
    - At least one entity in the semantic triple must match the target entity or be synonymous with the target entity. For example is the target entity is "pancreatic cancer" the triple "EFL2 -> negatively regulates -> NF-κB phosphorylation" should not be returned because it does not contain the target entity or a synonym of the target entity.
    - Return a numbered list of semantic triples that contain the target entity, with no additional commentary.
    - If no semantic triples are found, return "None" with no additional commentary.
    ###
    Example Input Format:
    Paragraph: Text
    Predicates:
    1. Predicate1
    2. Predicate2
    3. Predicate3
    Target Entity: Entity1
    ###
    Example Output Format:
    1. Entity1 -> Predicate1 -> Entity2
    2. Entity2 -> Predicate2 -> Entity1
    3. Entity4 -> Predicate3 -> Entity1
    """},
    {"role":"user", "content":
    """
    Paragraph: %s
    Predicates: %s
    Target Entity: %s
    """%(paragraph, predicates, entity_1)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def extract_entities(paragraph):
    messages = [{"role": "system", "content": """
    Task:
    You are a named entity recognition bot. Please extract all cell types, cell substructures, receptors, proteins, drugs, genes, biological processes, macromolecular complexes, small molecules, diseases, phenotypic feature, pathway, organism taxon, molecular activity, and anatomical structure contained within the provided text. Please also extract other biological entities. Return them as a numbered list with no additional commentary. Entities may contain overlapping words or phrases. Please be exaughstive in returning entities within the given categories. Do not return the the same entity twice.
    Example Output Format:
    1. Entity1
    2. Entity2
    3. Entity3"""},
    {"role":"user", "content":
    """
    %s
    """%(paragraph)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def choose_embedding_identifier(entity, labels):
    messages = [{"role": "system", "content": """
    Task:
    You will be given an entity, a list of potential labels including their descriptions. Please return the number of the closest matched label, with no additional formatting or commentary. Do not return the label itself.
    Example Input Format:
    Term: Text
    Labels:
    1. Label1
    2. Label2
    3. Label3
    Example Output Format:
    3
    """},
    {"role":"user", "content":
    """
    Term: %s
    Labels: %s
    """%(entity, labels)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def mech_paragraph_form(sentences):
    messages = [{"role": "system", "content": """Task: You are a helpful assistant. You will be given a series of short statements describing a mechanism of action. Please expand this to a full paragraph and return with no additional commentary."""},
    {"role":"user", "content":
    """
    %s
    """%(sentences)}
    ]
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)

def choose_embedding_match(possible_matches, term):
    messages = [{"role": "system", "content": """
    Task:
    You will be given a term, and list of descriptions. Your job is to select the description which best describes the term in a biomedical context from the given numbered list. Return only the number of the best equivalent description, with no additional commentary or formatting. 
    Example Input Format:
    Term: Text
    Descriptions:
    1. Description1
    2. Description2
    3. Description3
    Example Output Format:
    3
    """},
    {"role":"user", "content":
    """
    Term: %s
    Descriptions:
    %s
    """%(term, possible_matches)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def define_gpt(value, i=None):
    messages = [{"role": "system", "content": """Task:
You are a helpful assistant. You will be given a term and a brief description. Your job is to add supplemental information to the definition of the term by expanding on what is given. Do not return any additional commentary, only the expanded definition.
Example Input Format:
Term: Text
Definition: Text
Example Output Format:
Text"""},
    {"role":"user", "content":
    """
    %s
    """%(value)}
    ]
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    #print(i)
    response = str(completion.choices[0].message.content)
    return(response)

def contract_terms(terms):
    messages = [{"role": "system", "content": 
    """Task:
    You will be given a list of biological and chemical entities that contains duplicates with slightly different names. Your job is to deduplicate the list and return a list that contains only one version of a biological entity. Return a numbered list with no addional commentary or formatting.
    Example Input Format:
    1. Entity1
    2. Entity2
    3. Entity 1

    Example Output Format:
    1. Entity1
    2. Entity2
    """},
    {"role":"user", "content":
    """
    %s
    """%(terms)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def chain_of_thought(drug, disease):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant, who generates biological entities. 
           - Given a drug and the disease it treats, please return all biological and chemical entities through which they're connected as a numbered list.
           - Determine all biological and chemical entities that are involved in a mechanistic pathway where the drug treats the disease and return them as a numbered list. 
           - Entities include  proteins, organs, receptors, enzymes, chemical substances, small molecules, biological processes, biological pathways, cell types, organism taxons, gene families, molecular activities, macromolecular complexes, anatomical structures, and cellular components.
           - Do not return more than 12 entities. 
           - Less than 12 entities may be returned. 
           - Similar entities may be returned.
           - Return both the drug and disease in the entity list.
           - Return the most important entities at the top of the list and the least important at the bottom. 
           - Entities must be specifically named, for example instead of 'gene translation' name the specific gene translated.
           - Prioritize more specific entities, and entities more relevant to the mechanism of action of the drug treating the diease.
           ###
           Example Input Format:
           Drug: Text
           Disease: Text
           ###
           Example Output Format:
           1. Entity1
           2. Entity2
           3. Entity3
           """},
          {"role": "user", "content": "Drug: %s\nDisease: %s"%(drug, disease)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def chain_of_thought_3(drug, disease, entities, predicate_string, n=30):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           - You will create a knowledge graph consisting of nodes and edges, or predicates, that describes the relationship between the provided drug and disease.
           - The drug and disease will be given as input.
           - Nodes in the knowledge graph are biological and chemical entities and predicates describe the relationship between them.
           - All possible nodes and edges are provided in list form, but not all nodes and predicates must be used.
           - Provided nodes are listed in order of importance.
           - Return links as a numbered list, where every link consists of two nodes, and a predicate decribing the relationship between them.
           - Paths describe multiples node pairs and their links leading from one part of the graph to another.
           - The knowledge graph should contain at least one path going directly from the drug to the disease.
           - All returned links must contain a path to either the drug or disease.
           - Do not return more than %s links.
           - Do not return any additional commentary.
           ###
           Example Input Format:
           Drug: Text
           Disease: Text
           Predicates: 
           1. Predicate1
           2. Predicate2
           3. Predicate3
           Nodes:
           1. Node1
           2. Node2
           3. Node3
           4. Node4
           5. Node5
           6. Node6
           ###
           Example Output Format:
           1. Drug -> Predicate3 -> Node1
           2. Node1 -> Predicate1 -> Node2
           3. Node2 -> Predicate3 -> Disease
           4. Node3 -> Predicate2 -> Node1
           5. Node5 -> Predicate3 -> Node3
           """ %str(n)},
          {"role": "user", "content": "Drug: %s\nDisease: %s\nPredicates:\n%s\nNodes:\n%s"%(drug, disease, predicate_string, entities)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def chain_of_thought_2(paragraph, drug, disease, predicate_string, entity_string):
    prompt =[
          {"role": "system", "content": 
           """
           You will be given a drug, the disease it treats, a paragraph describing its mechanism of action, and a list of standardized predicates, and a list of biological entities. Your task is to construct a knowledge graph through a series of semantic triples to describe the mechanism of action.
- Each semantic triple consists of two nodes, and a predicate describing the relationship between them.
- No two semantic triples may contain the same two nodes. For example if the tripl 'Node1 -> Predicate1 -> Node2' is returned the triple 'Node1 -> Predicate3 -> Node2' may not be because it also contains Node1 and Node2.
- Predicates must be picked from the provided list.
- Entities must be picked from the provided list.
- Each node is a biological entity and can be a cell, a cellular component, a biological process, a receptor, a protein, a gene family, a chemical substance, a disease, a drug, a phenotypic feature, an anatomical structure, a macromolecular complex, organism taxon, or a pathway.
- You may return up to 50 semantic triples.
- Return the semantic triples as a 

Example Input Format:
Drug: Text
Disease: Text
Predicates:
1. Predicate1
2. Predicate2
3. Predicate3
Entities:
1. Node1
2. Node2
3. Node3
4. Node4
5. Node5
Mechanism: Text

Example Output Format:
1. Drug -> Predicate3 -> Node1
2. Node1 -> Predicate1 -> Node2
3. Node2 -> Predicate3 -> Disease
4. Node3 -> Predicate2 -> Node1
5. Node5 -> Predicate3 -> Node3
           """},
          {"role": "user", "content": "Drug: %s\nDisease: %s\nPredicates:\n%s\nEntities:\n%s\nMechanism:\n%s"%(drug, disease, predicate_string, entity_string, paragraph)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def alternate_mechanism_one_shot(drug, disease, predicate_string):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant, who creates knowledge graphs. Given a drug and the disease it treats, please return all  mechanisms of action as a series of numbered steps. 
            Forming a mechanims of action:
            Determine all biological and chemical entities that consistute a mechanistic pathway where the drug treats the disease, including all proteins, organs, receptors, enzymes, chemical substances, small molecules, biological processes, biological pathways, cell types, organism taxons, gene families, molecular activities, macromolecular complexes, anatomical structures, and cellular components. Be specific and expansive without going in a circle.
           Translate these entities into a stepwise mechanism of action.
           Each step should consist of two node terms, connected by a predicate. The second node term for a step is always the first node term for the subsequent step. The first node term of the first step is always the drug name, and the second node terms of the final step is always the disease name. The names of the node terms are case sensitive.
           The predicates that may be used to describe the relationship between the node terms will be provided. Please pick a single predicate from the given list. Predicates should be chosen such that each step can stand alone and be true.
           Node terms are biological or chemical entities, and can be proteins, organs, receptors, enzymes, chemical substances, small molecules, biological processes, biological pathways, cell types, organism taxons, gene families, molecular activities, macromolecular complexes, anatomical structures, cellular components, phenotypic features, diseases or drugs. Node terms cannot contain descriptive terms such as "lower", "higher", "decrease", "increase", "reduced", "level" or any other similar quantity or measurement modifiers. Node terms contain no indication of biological function. Node terms cannot contain predicates or verbs.
           You may return up to 15 steps describing the mechanism, and do not return any additional commentary.
            Forming the overall knowledge graph:
            You may return multiple mechanisms for the drug treating the disease. Each mechanism must have at least one distinct node. Seperate mechanims by a newline character.

           Example Input Format:
           Drug: Text
           Disease: Text
           Predicates: 
           Predicate1
           Predicate2
           Predicate3
           Example Output Format:
           1. Drug -> Predicate3 -> Entity1
           2. Entity1 -> Predicate1 -> Entity2
           3. Entity2 -> Predicate3 -> Disease

           1. Drug -> Predicate3 -> Entity1
           2. Entity1 -> Predicate2 -> Entity3
           3. Entity3 -> Predicate1 -> Entity4
           4. Entity4 -> Predicate1 -> Disease
           """},
          {"role": "user", "content": "Drug: %s\nDisease: %s\tPredicates: %s"%(drug, disease, predicate_string)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def alternate_ground_predicates(term1, term2, predicate, messages, category_string):
    prompt = [{"role": "user", "content": """
    Task:
    You will be given two biological or chemical entities, and predicate describing the relationship between them. Your job is to select the predicate from the given numbered list of the best equivalent to the original predicate. Return only the number of the best equivalent predicate, with no additional commentary or formatting. 
    Example Input Format:
    Entity 1: Text
    Entity 2: Text
    Predicate : Text
    1. Predicate1
    2. Predicate2
    3. Predicate3
    Example Output Format:
    3
    Input:
    Entity 1: %s
    Entity 2: %s
    Predicate %s
    %s"""%(term1, term2, predicate, category_string)}]
    messages.extend(prompt)

    completion = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def test_node_grounding(database, node_name):
    prompt =[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": """What is the structured identifier for '%s' in the %s database? Please return ten closest identifiers as a list in the following format  with no additional commentary: ["identifier1", "identifier2", "identifier3", ...]. If the term cannot be found within the database, please return the ten identifiers of the most closely related entities. Please retunr these as a list in the following format with no additional commentary: ["identifier1", "identifier2", "identifier3", ...]."""%(node_name, database)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def alternate_pick_grounding(entity, all_matches):
    prompt =[
          {"role": "system", "content": 
           """
           You are a helpful assistant. You will be given a list of dictionaries, where each dictionary contains a grounding result for a biological or chemical entity. You will also be given a target biological entity. Please return the number of the dictionary that best represents the target biological entity. Please return the index with no additional formatting or commentary.
           """},
          {"role": "user", "content": "%s\n%s"%(entity, all_matches)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)



def alternate_equivalency_check(term1, term2):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant. You will be given two biological or chemical entities. Please determine whether or not they are equivalent and can be used interchangeably to mean the same thing. Return "yes" if they are equivalent and "no" if they are not equivalent. Do not return additional commentary, punctuation, or formatting.
           Example Input Format:
           Term 1: Text
           Term 2: Text
           Example Output Format:
           yes
           """},
          {"role": "user", "content": "Term 1: %s\nTerm 2: %s"%(term1, term2)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def alternate_categorize_node(node, messages, category_string):
    prompt = [{"role": "user", "content": """
    Task:
    You are a helpful assistant. Which of these statements best describes the category of %s? Be as specific as possible, and return only the number associated with the best statement, with no additional commentary.
    Example Input Format:
    1. Statement 
    2. Statement
    3. Statement
    Example Output Format:
    3
    Input:
    %s"""%(node, category_string)}]
    messages.extend(prompt)

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def parse_html(html, entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant. You will be given a body of html. Please parse the HTML and summarize the contents of the page in paragraph form, including all headers, titles, unique identifiers and detailed information relating to the biological or chemical entity it describes. Ignore information related to the formatting of the webpage itself. Do not return additional commentary, punctuation, or formatting.
           """},
          {"role": "user", "content": "%s"%(html)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def alternate_extract_identifiers(text, entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant. Given a paragraph of text and the name of a biological or chemical entity, please extract the unique identifier associated with it. Return the unique identifier as a string with no additional commentary or formatting. If no identifier can be found, or the boyd of text does not match the entity, please return "not found".
           """},
          {"role": "user", "content": "Paragraph: %s\nEntity: %s"%(text, entity)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def alternate_url_ground_react(entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant. You will be given a biological entity term, and your task is to return a the Reactome (REACT) URL for the entity term. Do not return any additional commentary, only the URL. If no URL can be identified, please return "not found".
           """},
          {"role": "user", "content": "%s"%(entity)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def find_urls(entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You will be given a biological entity. Please describe and define what that biological entity is. Next, please find an return a list of urls corresponding to databases which contain structured identifiers for the following biological entity. Example databases include PubChem, DrugBank, Human Phenotype Ontology, Gene Ontology, MeSH, CHEBI, and MONDO. If no urls can be found, return an empty list.
           Example Input Format:
           Entity
           Example Output Format:
           ["url1", "url2", "url3"]
           """},
          {"role": "user", "content": "%s"%(entity)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def choose_paths(string_paths, mechanism, messages=[]):
    prompt = [{"role": "user", "content": """
    Task:
    You are a helpful assistant. Which of these statements best describes the category of %s? Be as specific as possible, and return only the number associated with the most accurate summary of the mechanism, with no additional commentary.
    Example Input Format:
    1. Statement 
    2. Statement
    3. Statement
    Example Output Format:
    3
    Input:
    %s"""%(target, statements)}]
    messages.extend(prompt)

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)



def check_type_grounding(statements, target, mechanism, messages=[]):
    prompt = [{"role": "user", "content": """
    Task:
    You are a helpful assistant. Which of these statements best describes the category of %s? Be as specific as possible, and return only the number associated with the best statement, with no additional commentary.
    Example Input Format:
    1. Statement 
    2. Statement
    3. Statement
    Example Output Format:
    3
    Input:
    %s"""%(target, statements)}]
    messages.extend(prompt)

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def additional_grounding(statements, target, messages=[]):
    prompt = [{"role": "user", "content": """
    Task:
    You are a helpful assistant. Which of these statements is the closest synonym to %s? Return only the number associated with the best statement, with no additional commentary.
    Example Input Format:
    1. Statement 
    2. Statement
    3. Statement
    Example Output Format:
    3
    Input: 
    %s"""%(target, statements)}]
    messages.extend(prompt)

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def describe_relationship(sub, obj):
    prompt =[
          {"role": "system", "content": 
           """
           You are a helpful assistant.
           """},
          {"role": "user", "content": "What is the relationship between '%s' and '%s', in a biochemical context, if any?"%(sub, obj)}
      ]

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def testy_test(statements, sub, obj, messages):
    prompt = [{"role": "user", "content": """
    Task:
    You are a helpful assistant. Which of these statements best describes the relationship between %s and %s? Return only the number associated with the best statement, with no additional commentary. If the relationship cannot be clearly and readily described by the statements present, please return "no relationship". It is better to return "no relationship" than an inaccurate description.
    Example Input Format:
    1. Statement 
    2. Statement
    3. Statement
    Example Output Format:
    3
    Input: 
    %s"""%(sub, obj, statements)}]
    messages.extend(prompt)

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=messages
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def grammatical_check(prompt):
    completion = client.chat.completions.create(
    model="gpt-4",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def clean_up_entities(entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You will be given a biological or chemical entity and your job is to determine if the entity contains any sub-entities. All sub-entities must be biological or chemical entities. Please return sub-entities as a comma-seperated string with no leading white spaces, additional formatting or commentary.
           Example Input:
           Does 'G-coupled protein receptor' contain any sub-entites?
           Example Output:
           G-coupled protein,protein,receptor,G-coupled protein receptor
           """},
          {"role": "user", "content": "Does '%s' contain any sub-entities?"%(entity)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def extract_mech_path(paragraph, drug, disease):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           Given a paragraph of information, a drug, and a disease state, please explain in paragraph form the mechanism of action through which the drug treats the disease. Use specific and complete biological and chemical terminology whenever possible. Supplement the given information if needed. Be as bioloigcally specific and detailed as possible and provided background explanations about the entities involved in the mechanism.
           Additionally, please return a complete list of the bioloigical or chemical entities, responses, or terms, including sub-entities and sub-terms, that are a part of this mechanistic path seperated by commas.Be as comprehensive as possible. Please return the response in the format specified with no  additional commentary.

          Example Input:
          Paragraph:NSAIDs such as ibuprofen work by inhibiting the cyclooxygenase (COX) enzymes, which convert arachidonic acid to prostaglandin H2 (PGH2). PGH2, in turn, is converted by other enzymes to several other prostaglandins (which are mediators of pain, inflammation, and fever) and to thromboxane A2 (which stimulates platelet aggregation, leading to the formation of blood clots). Like aspirin and indomethacin, ibuprofen is a nonselective COX inhibitor, in that it inhibits two isoforms of cyclooxygenase, COX-1 and COX-2. Based on this mechanism, headaches are treated by ibuprofen. It has been know to also treat inflammation.
          Drug:ibuprofen
          Disease:headaches

          Example Output:
          Mechanism:ibuprofen inhibits cyclooxygenase (COX) enzymes. Specifically, ibuprofen inhibits COX-1 and COX-2 enzymes. COX enzymes convert arachidonic acid to prostaglandin H2 (PGH2). PGH2 is converted to other prostaglandins, thromboxanes, and prostacyclins. Prostaglandins, thromboxanes, and prostacyclins are implicated in causing pain, inflammation, and fever. Thus, ibuprofen reduces prostaglandins to prevent pain associated with headaches.
          Relevant Entities:ibuprofen,cyclooxygenase (COX) enzymes,enzymes,COX-1 enzyme,COX-2 enzyme,arachidonic acid,prostaglandin H2 (PGH2),H2,prostaglandin,prostaglandins,thromboxanes,prostacyclins,inflammation,fever,pain
          """},
          {"role": "user", "content": "Paragraph:%s\nDrug:%s\nDisease:%s"%(paragraph, drug, disease)}
      ]


    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


