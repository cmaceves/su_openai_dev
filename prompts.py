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
You are a helpful assistant. You will be given a term and a brief description. Your job is to add supplemental information to the definition of the term by expanding on what is given. Do not return any additional commentary, only the expanded definition
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
    response = str(completion.choices[0].message.content)
    print(i)
    return(response)

def lemmatize(term):
    messages = [{"role": "system", "content": """Task:
You are a helpful assistant. You will be given a scientific term or phrase. Your job is to lemmatize it such that you return its singular form. Do not return any additional commentary or formatting.
Example Input Format:
Cells
Example Output Format:
Cell"""},
    {"role":"user", "content":
    """
    %s
    """%(term)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)


def choose_wikidata_match(term, page_strings):
    messages = [{"role": "system", "content": """
    Task:
    You will be given a term, and list of webpages and their descriptions. Your job is to select the webpage which best matches the term in a biomedical context from the given numbered list of webpages and descriptions of their content. Some webapges may be missing descriptions. Return only the number of the best equivalent webpage, with no additional commentary or formatting. 
    Example Input Format:
    Term: Text
    Webpages:
    1. Page1 - Description1
    2. Page2 - Description2
    3. Page3
    Example Output Format:
    3
    """},
    {"role":"user", "content":
    """
    Term: %s
    Webpages:
    %s
    """%(term, page_strings)}]

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0
    )
    response = str(completion.choices[0].message.content)
    return(response, messages)

def alternate_mechanism_one_shot(drug, disease, predicate_string):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant, who creates knowledge graphs. Given a drug and the disease it treats, please return the mechanism of action as a series of numbered steps. 
           Determine the important biological and chemical entities used to describe the mechanistic pathway where the drug treats the disease.
           Translate these entities into a stepwise mechanism of action. Not every entity determined to be important must be used in this mechanism, be brief when possible.
           Each step should consist of two node terms, connected by a predicate. The second node term for a step is always the first node term for the subsequent step. The first node term of the first step is always the drug name, and the second node terms of the final step is always the disease name. The names of the node terms are case sensitive.
           The predicates that may be used to describe the relationship between the node terms will be provided. Please pick a single predicate from the given list. Predicates should be chosen such that each step can stand alone and be true.
           Node terms are biological or chemical entities, and can be proteins, organs, receptors, enzymes, chemical substances, small molecules, biological processes, biological pathways, cell types, organism taxons, gene families, molecular activities, macromolecular complexes, anatomical structures, cellular components, phenotypic features, diseases or drugs. Node terms cannot contain descriptive terms such as "lower", "higher", "decrease", "increase", "reduced", "level" or any other similar quantity or measurement modifiers. Node terms contain no indication of biological function. Node terms cannot contain predicates or verbs.
           You may return up to 10 steps describing the mechanism, and do not return any additional commentary.

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
          {"role": "user", "content": "What is the structured identifier for '%s' in the %s database? Please return the identifier with no additional commentary, or 'not found' if no identifier can be found."%(node_name, database)}
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

def alternate_entity_expansion(entity, type_restriction, messages):
    prompt = [
          {"role": "user", "content": 
           """
           Task:
           You are a helpful assistant. You will be given a biological entity term and it's category, and your task is to return a list of synonyms. This list of synonyms should be an expanded list of terms that may be used to ground the same entity, assign an identifier, in a downstream task. Any synonym returned must fall within the same categorization as the original entity. Synonyms should be returned as a numbered list seperated with newlines. Do not return any additional commentary.
           Example Input Format:
           Entity: Entity
           Category: Entity Category
           Example Output Format:
           1. Synonym1
           2. Synonym2
           3. Synonym3
           """},
          {"role": "user", "content": "Entity: %s\nCategory: %s"%(entity, type_restriction)}
      ]
    messages.extend(prompt)
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=prompt,
    temperature = 0
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def alternate_mechanism(drug, disease):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant, who creates knowledge graphs. Given a drug and the disease it treats, please return the mechanism of action as a series of numbered steps. 
           Determine the important biological and chemical entities used to describe the mechanistic pathway.
           Translate these entities into a stepwise mechanism of action. Not every entity determined to be important must be used in this mechanism, be brief when possible.
           Each step should consist of two node terms, connected by a predicate. The second node term for a step is always the first node term for the subsequent step. The first node term of the first step is always the drug name, and the second node terms of the final step is always the disease name. The names of the node terms are case sensitive.
           Node terms are biological or chemical entities, and can be proteins, organs, receptors, enzymes, chemical substances, small molecules, biological processes, biological pathways, cell types, organism taxons, gene families, molecular activities, macromolecular complexes, anatomical structures, cellular components, phenotypic features, diseases or drugs. Node terms cannot contain descriptive terms such as "lower", "higher", "decrease", "increase", "reduced", "level" or any other similar quantity or measurement modifiers. Node terms contain no indication of biological function. Node terms cannot contain predicates or verbs.
           You may return up to 10 steps describing the mechanism, and do not return any additional commentary.

           Example Input Format:
           Drug: Text
           Disease: Text
           Example Output Format:
           1. Drug -> Predicate -> Entity1
           2. Entity1 -> Predicate -> Entity2
           3. Entity2 -> Predicate -> Disease
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


def describe_accuracy(statement):
    prompt =[
          {"role": "system", "content": 
           """
           You are a helpful assistant.
           """},
          {"role": "user", "content": "Is the statement '%s' accurate? Please respond with a 'yes' or 'no', and if the statement is only partically accurate respond with 'no'. Do not return any additional commentary or formatting."%(statement)}
      ]

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=prompt
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

def synonym_context(word1, word2, paragraph):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant. You will be given a paragraph, a word of interest, and a synonym for the word of interest. Your job is to determine whether or not the synonym means the same thing as the word of interest in the context of the paragraph. If the word and the synonym mean the same thing in the context of the pargraph, return "yes" otherwise return "no" in lowercase with no punctuation. Do not return any additional commentary or formatting.
           Example Input Format:
           Paragraph:Insert paragraph here.
           Word of interest: Insert word here.
           Synonym: Insert synonym here.

           Example Output Format:
           yes
           """},
          {"role": "user", "content": "Paragraph:%s\nWord of interest:%s\nSynonym:%s"%(paragraph, word1, word2)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def synonym_prompt(entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are a helpful assistant. You will be provided a biological entity. Return each synonym on a newline as a numbered list with no additional commentary or formatting.
           Example Output Format:
           1. Synonym1
           2. Synonym2
           3. Synonym3
           """},
          {"role": "user", "content": "What are all synonyms for '%s'?"%(entity)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def test_prompt(page, mechanism):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You are trying to improve a description of a mechanism of action of a drug. To do so you must determine which Wikipedia database pages will provide relevant additional information. Given the title of a Wikipedia database page, and the known description mechanism of action, determine whether or not the Wikipedia database page can improve our knowledge. Please return your answer as either a "yes", meaning the page will provide relevant information or "no" meaning that the page will not provide relevant information. Do not return any additional commentary or formatting.
           """},
          {"role": "user", "content": "Wikipedia Database Page Title:%s\nMechanism:%s"%(page,mechanism)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def summarize_add_info(text, mechanism):
    prompt =[
          {"role": "system", "content": 
           """
          Given a paragraph of text, and a mechansims by which a drug treats a disease, please summarize the paragraph of text while retaining all information possibly relevant to the mechanism.
          """},
          {"role": "user", "content": "Paragraph:%s\nMechanism:%s"%(text,mechanism)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
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

def summarize_text(text):
    completion = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": 
          """
          You are a helpful assistant, used to summarize scientific text. You will be given a chunk of text and asked to summarize it while retaining all named scientific entities. Please return the summary with no additional commentary or formatting.
          """},
          {"role": "user", "content": 'Summarize the following text: %s' %(text)}
      ]
    )
    resp = str(completion.choices[0].message.content)
    return(resp)
