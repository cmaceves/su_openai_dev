"""
Skeptic - Asks questions about the given informatin.
Librarian - Looks up information to answer specific questions.
Formatter - Formats the response into the given schema.
Reformatter - Summarized additional information along with original information.
Language Arts Teacher - Take the text and return subject-object-predicate relationships.
Older Brother - Asks if the subject object predicate reltionships are ACTUALLY correct.
Detective Query - Finds alternative IDs the the same things.
Journal Editor - Given the subject, object, predicate relationships, pull scientific sources.
"""
import os
import sys
import ast
import json
import requests
import numpy as np
import html_to_json
from openai import OpenAI
from bardapi import Bard
from normalize import get_normalizer #taken from Benchmarks
client = OpenAI()
def parse_schema_response(schema):
    """
    Parameters
    -----------
    schema : str
        The known information on the given schema in string form.
    return_dict : dict
        The schema parsed as a dictionary
    """
    try:
        return_dict = ast.literal_eval(schema)
    except:
        print(schema)
        ast.literal_eval(schema)
        sys.exit(0)
    return(return_dict)

def equivalency(content):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """You're a helpful curator of scientific information. You will be given a list of identifiers in the following format: ["Identifier1", "Identifer2", "Identifier3"].
          It is your job to determine if these identifiers all refer to the same thing, and are euqivalent.Please return your answer as "True" if all the identifiers are equivalent and "False" if all the identifiers are not equivalent. If you cannot determine the answer, return "False". Do not return and additional information or commentary."""},
          {"role": "user", "content":"%s"%(content)}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def language_arts_tutor_query(word, part_of_speech):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """You're a helpful tutor. You will be given a phrase and a part of speech, and your job is to determine if the given phrase contains that part of speech, and if so count the number of that part of speech present. Please return "True" if it does contain the given part of speech and "False" if it does not match the given part of speech.Pronouns are not considered nouns. For example you'll be asked for example if:
           noun : dog? 
           to which you would reply:
           True
           If you cannot determine the part of speech, please reply "False."  """},
          {"role": "user", "content": "%s : %s?"%(word, part_of_speech)}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def language_arts_teacher_query(content, predicates):
    """
    The language arts teacher decides to summarize the data populating the schema into subject-predicate-obejct triplets. 
    """
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """
          You're a helpful curator of scientific information, helping to extract relationships between entities from text. Given the following summary, please give a list of all subject, predicate, and object tuples which describe the relationships in the following text. 
          A noun is a word for a person, place, thing, or idea. The verb in a sentence expresses action or being. A predicate is the part of the sentence containing the verb, and should not include any nouns. Every subject, predicate and object tuple can contain only a single verb. Both the subject and the object values should contain nouns, and the predicate should capture the relationship between the subject and the object using a verb. No verbs should be included in the subject phrase or the object phrase. 
          Predicates that are acceptable to be be returned as a part of the answer can be found in this list: %s. Do not return a predicate values that does not appear in this list. If no accurate predicate can be found return "Not Specified". Object and subject values should not contain more than one noun, although nouns may be multiple words.
          Return "Not Specified" if no subject, predicate, object tuples can be extracted from the text or one of the three parts of speech is missing. Do not return tuples containing less than three values. All subject, object, and predicate strings should be individually enclosed in double quotations. Return the data in the following json parsable format, with no additional content or commentary.:
           [("Subject", "Predicate", "Object"), ("Subject", "Predicate", "Object"), ("Subject", "Predicate", "Object"), ("Not Specified", "Predicate", "Object")]""" %predicates},
          
          {"role": "user", "content": "Do not use predicates in the answer that do not appear in this list: %s. Please extracts subject, predicate, object tuples from the following text: %s" %(predicates, content)}
      ]
    )
    return(str(completion.choices[0].message.content))

def reformatter_query(content, field):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """You are a curator of scientific information, helping scientists understand drug mechanisms of action, chemistry, associated pathways, and side effects. Please summarize the following text, preserving all unique subjects, objects, and predicates:
"""},
          {"role": "user", "content": "Please summarize the following: %s as it relates to %s." %(content, field)}
      ]
    )

    return(str(completion.choices[0].message.content))

def older_brother_query(subject, predicate, objec):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """You are a highly skeptical curator of scientific information, and you have been given a statement containing a subject, a predicate, and an object. Your job is to determine whether this relationship is real or fabricated. Respond with a 'yes' if you think this is a true relationship, and 'no' if you think this is a fabricated relationship.
"""},
          {"role": "user", "content": "Tell, me using a 'yes' or 'no', is this relationship '%s %s %s' true?" %(subject, predicate, objec)}
      ]
    )
    return(str(completion.choices[0].message.content))

def skeptic_query(original_content, new_content, focus_predicates):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """You are a curator of scientific information, and your job is to make sure all information provided is correct, and to create further questions about the information in the given schema to expand the knowledge in the schema. 
          In particular, please make sure to ask questions about fields marked as 'Not Specified' or blank fields '', however ask questions about any field where more knowledge could be added. 
          All questions should use the following predicates: %s to relate the %s to the field title.
          Please be as specific as possible. Please return your answer in a json format and only json format, where each key is a field in the original schema and every value is a list of questions to be asked, if any. Do not return any additional information or commentary. An example of output format is given below:
          {
          'Field': ['Question', 'Question2']
          'Field2': ['Question']
          }
           """ %(original_content, focus_predicates)},
        {"role": "user", "content": "The original schema was %s and the response schema was %s. What additional questions do you have?" %(original_content, new_content)}
      ]
    )
    return(str(completion.choices[0].message.content))

def formatter_query(content, original_schema):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": """You are a curator of scientific information, helping scientists understand drug mechanisms of action, chemistry, associated pathways, and side effects. You will be given a standardized identifier. Please find information related to the given identifier, and format it into the given dictionary schema, using 'Not Specified' to fill fields where information is uknown. Enclose each response with double quotation characters. If there are multiple responses, please write them each as individual and complete sentences. Do not exclude any of the fields below, and do not return any additional information: 
        %s""" %original_schema},
        {"role": "user", "content": "%s" %content}
      ]
    )
    return(str(completion.choices[0].message.content))

def librarian_query(known, content, original):
    """
    Paramters 
    ---------
    known : str
    content : str
    """
    if known == "Not Specified":
        content = "Right now we have no information about %s. Could you tell me "%original + content
    else:
        content = "Right now we know %s about %s. Could you tell me "%(known, original) + content
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": """You are a librarian of scientific information, and your job is to answer the questions to address gaps in the known information. If you cannot answer the question accurately, please response 'No Answer'."""},
        {"role": "user", "content": "%s"%content}
      ]
    )
    return(str(completion.choices[0].message.content))

def text_tense(content):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": """You are a helpful assistant used to reformat text. Please take the input text, and format it to be enitrely in the present tense using an active voice. Return the present tense text without any additional commentary"""},
        {"role": "user", "content": "%s"%content}
      ]
    )
    response = str(completion.choices[0].message.content)

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": """You are a helpful assistant used to reformat text. Please take the input text, and format it so that it contains no pronouns. Whenever you find a pronoun, replace it with the original noun. Return the reformatted text without any additional commentary."""},
        {"role": "user", "content": "%s"%response}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def detective_query(content):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": 
        """
        You are a helpful assistant, who gives the user a list of possible websites from databases to explore. Give a text input, please return a json parseable list of url values where identiers for the nouns in the text can be found. Do not return any additional commentary. Do not return and url values which could belong to a different entity. Example output:
         ["Url1", "Url2", "Url3"] """},
        {"role": "user", "content": "Please list websites to assist in the identification and curation of %s."%content}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def html_finder(content):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": 
        """
        You are a helpful assistant, who specializes in programming html, javascript, and css for web applications, and your goal is to capture the contents of the page without any programmatic elements. Please look at this chunk of text and remove all html keywords and formatting, all css content,  and all javascript script, keywords and formatting. Remove all newline characters, url values, and css styling components. Return the text in paragraph form without any additional commentary. 
         """},
        {"role": "user", "content": "Please reformat this text: %s" %content}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def content_finder(content, value):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": 
        """
        You are a helpful assistant, who specializes in finding identifiers for entities given a body of text from a database. Given a body of text, find the identifier correpsoniding to the given value. Return a dictionary, where the key is the name of the database and the value is a list of the identifiers. Use the example format below, without any additional commentary:
        {"Database": ["Indentifier1", "Identifier2", "Indentifier3"]}
         """},
        {"role": "user", "content": "Please find identifiers for this value: %s in this text: %s" %(value, content)}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def main(): 
    print(get_normalizer(['cid2200']))
    sys.exit(0)
    urls = detective_query("Antazoline")
    urls = ast.literal_eval(urls)
    print(urls)
    for url in urls:
        resp = requests.get(url)
        print('h')
        content = resp.content
        ncontent = ""
        for i in range(0, len(content), 2000):
            if i > 10000:
                break
            con = html_finder(content[i:i+2000])
            ncontent += " " + con
        ncontent = ncontent.lower().strip()
        ids = content_finder(ncontent, "Antazoline")
        db = list(ast.literal_eval(ids).keys())
        print(ids, db)
        sys.exit(0)
    sys.exit(0)
    
    original_identifier = "MESH:D000865"
    predicate_json = "predicates.json"
    original_schema = {"drug":"",
                       "detailed mechanism of action":"",
                       "protein":"",
                       "disease":"",
                       "receptor":"",
                       "chemical substance":"",
                       "anatomical structure": "",
                       "organism":"",
                       "cell": "",
                       "gene family":"",
                       "pathway":"",
                       "molecular activity":"",
                       "cellular component":""
                       }

    #load canon predicates from the biolink model yaml
    with open(predicate_json, 'r') as jfile:
        data = json.load(jfile)
    focus_predicates = data['predicates']

    #identifier to text
    all_identifiers = get_normalizer([original_identifier])
    all_identifiers = list(np.unique([x.lower() for x in all_identifiers]))
    if len(all_identifiers) > 1:
        print("multiple labels")
        sys.exit(0)
    original_content = all_identifiers[0]        
    #format some initial text related to the fields provided in the schema
    schema_response = formatter_query(original_content, original_schema)
    return_dict = parse_schema_response(schema_response)
    #retrieve relevant text and ask questions about it
    for i in range(1):
        print(i)
        response = skeptic_query(original_content, schema_response, focus_predicates)
        question_dict = parse_schema_response(response)
        for field, resp in question_dict.items():
            #this needs to be generalizable
            if field == "drug":
                continue
            content_gather = []
            for question in resp:
                additional_info = librarian_query(return_dict[field], question, original_content)    
                content_gather.append(additional_info.strip().replace("\n",""))
            if len(content_gather) == 0:
                continue
            content_gather = ". ".join(content_gather) + " " +return_dict[field]
            summarizer = reformatter_query(content_gather, field)
            return_dict[field] = summarizer
        schema_response = json.dumps(return_dict)
    
    print("begin curating data") 
    return_dict = parse_schema_response(schema_response)
    real_triplets = []

    #format it into triplets and retrieve alternative identifiers    
    for key, value in return_dict.items():
        if value == "Not Specified":
            continue
        if key == 'drug':
            continue
        print(value)
        rf = text_tense(value)
        print(rf)
        triplets = language_arts_teacher_query(rf, focus_predicates)

        if triplets.lower() == 'not specified':
            continue
        triplet_list = ast.literal_eval(triplets)
        for triplet in triplet_list:
            if len(triplet) < 3:
                continue
            subject = triplet[0].lower().strip()
            predicate = triplet[1].lower().strip()
            objec = triplet[2].lower().strip()     

            #does this contain the desired part of speech
            correct_subject = language_arts_tutor_query(subject, "noun")
            correct_object = language_arts_tutor_query(objec, "noun")
            correct_predicate = language_arts_tutor_query(predicate, "verb")
            if not bool(correct_subject) or not bool(correct_predicate) or not bool(correct_object):
                continue      

            #is this factually accurate
            ground_truth = older_brother_query(subject, predicate, objec)
            if 'yes' in ground_truth.lower():                
                real_triplets.append((subject, predicate, objec))
                print((subject, predicate, objec))

                #TODO work on the detective query bit...          
                other_subject_identifiers = detective_query(subject)
                print(other_subject_identifiers)
                continue
                """
                real_subject_identifiers = []
                if other_subject_identifiers != "No Answer":
                    other_subject_identifiers = ast.literal_eval(other_subject_identifiers)
                    same = equivalency(other_subject_identifiers)
                    if not bool(same):
                        continue
                    norm = get_normalizer(other_subject_identifiers)
                    for key, value in norm.items():
                        real_subject_identifiers.extend([key, value])
                other_object_identifiers = detective_query(objec)
                real_object_identifiers = []
                if other_object_identifiers != "No Answer":
                    other_object_identifiers = ast.literal_eval(other_object_identifiers)
                    same = equivalency(other_object_identifiers)
                    if not bool(same):
                        continue
                    norm = get_normalizer(other_object_identifiers)
                    for key, value in norm.items():
                        real_object_identifiers.extend([key, value])
                if len(real_object_identifiers) == 0:
                    continue
                print("other subject id", other_subject_identifiers)
                print("other obj id", other_object_identifiers)
                print("real obj id", real_object_identifiers)
                print("real subj id", real_subject_identifiers)
                print(triplet)
                process(real_object_identifiers[0], focus_predicates)
                sys.exit(0)
                """
    with open("subobjpred.json","w") as jfile:
        json.dump(real_triplets, jfile)

if __name__ == "__main__":
    main()
