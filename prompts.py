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
