import sys
import ast
import json
import pandas as pd
from openai import OpenAI
client = OpenAI()

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
    model="gpt-4-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def testy_test(statements, sub, obj, messages):
    prompt = [{"role": "user", "content": """
    Task:
    You are a helpful assistant. Which of these statements best describes the relationship between %s and %s? Return only the number associated with the best statement, with no additional commentary. If the relationship cannot be clearly and readily described by the statements present, please return "no relationship". It is better to retrun "no relationship" than an inaccurate description. The relationships between subjects and objects are characterized by verbs, and the directions of these statments are criticial. For example just because "subject verbs object" is true does not mean that "object verbs subject" is also true.
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


def test_prompt_2(mechanism, entity):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           You will be given a biological or chemical entity and the text it appears in. Certain entities contain more specific synonyms, please return all such synonyms. Be as comprehensive and specific as possible. Return each synonym on a newline in the following format, with no additional commentary or formatting.
           Output Format:
           Synonym1
           Synonym2
           Synonym3
           """},
          {"role": "user", "content": "Text:%sEntity:%s"%(mechanism, entity)}
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
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def sentence_pos_replacement(sentence, pos):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": 
           """
           Task:
           You will be given a paragraph, and the parts of speech for every word in that paragraph in the same order. Please remove all adjectives and replace all pronouns with their corresponding proper nouns. Do not add any additional commentary or spacing.

           Example Input 1:
           Paragraph- John went to a crowded grocery store. He bought apples. He then ate the green apples.
           Parts of Speech- proper noun;verb past tense;infinite marker;determiner;verb past participle;noun;noun;personal pronoun;verb past tense;noun plural;personal pronoun;adverb;verb;determiner;adjective;noun plural;

           Example Output 1:
           John went to a grocery store. John bought apples. John then ate the apples.
           """},
          {"role": "user", "content": "Paragraph- %s\nParts of Speech- %s"%(sentence, pos)}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def entity_context(text, term):
    completion = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": 
          """
          Task:
          You will be given a paragraph of text, and asked to identify the species context of a biomedical word with the text. Please respond with "human", "mouse", or "not identified" in the case where a species cannot be determined. Return no addtional commentary.
          Input:
          Protein X was found to be upregulated four-fold in patients with syndrome Y.
          Output:
          human
          """},
          {"role": "user", "content": 'In what species context is %s used within this paragraph: %s' %(term, text)}
      ]
    )
    resp = str(completion.choices[0].message.content)
    return(resp)


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

def similar_terms(term1, term2):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": 
          """
          You are a helpful assistant used to determine if two words or phrases mean the same thing. Please answer the following question with a "yes" or a "no" in lowercase with no additonal commentary or punctuation.
          Example Output:
          yes
          """},
          {"role": "user", "content": 'Is %s the same as %s' %(term1, term2)}
      ]
    )
    resp = str(completion.choices[0].message.content)
    return(resp)

def entity_categorization(questions):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": 
          """
          Task:
          You are a helpful assistant used for entity categorization. You will be asked a series of questions, and will need to respond to each question individually with a "#.yes" or "#.no" in lowercase letters with no additional commentary. The "#" character in the response indicates the order of questions asked and should be a numeric value.

          Input Format:
          1.Is Entity a Category1?
          2.Is Entity a Category2?
          3.Is Entity a Cateogry3?

          Output Format:
          1.no
          2.no
          3.yes

          Example Input:
          1.Is a cat a dog?
          2.Is a cat a pet?
          3.Is a dog a pet?
          4.Is a snake a dog?

          Example Output:
          1.no
          2.yes
          3.yes
          4.no

          Example Input:
          1.Is a sock a clothing?

          Example Output:
          1.yes
          """},
          {"role": "user", "content": '%s' %(questions)}
      ]
    )
    resp = str(completion.choices[0].message.content)
    return(str(completion.choices[0].message.content))

def other_entity_query(paragraph, entity_string):
    prompt = [
          {"role": "system", "content": 
          """
          You are a helpful assistant for entity extraction. Given a paragraph, and a list of entity categories with definitions, please return a Pythonic dictionary where each key is an entity contained in the paragraph and each associated value is its category. Do not return any additional commentary.
          
          Input Format:
          Paragraph: Text sentences describing some biolgicial thing.
          Entities:
          Entity category 1:entity definition.
          Entity category 2:entity definition.

          Output Format:
          {"Entity1": "category", "Entity2": "category"}
          """},
          {"role": "user", "content": "Paragraph:%s\nEntities:%s" %(paragraph, entity_string)}
      ]

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=prompt)
    return(str(completion.choices[0].message.content), prompt)



def entity_query(paragraph):
    prompt = [
          {"role": "system", "content": 
          """
          You are a helpful assistant used for entity extraction. Given a paragraph and a list of categories please return the biological, medical, or chemical entities or processes in the paragraph in a Pythonic list form with no additional commentary.
          Output Format:
          ["Term1", "Term2", ...]
          
          """},
          {"role": "user", "content": "What are the biological, medical, or chemical entities or processes in the following paragraph: %s" %(paragraph)}
      ]

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=prompt)
    return(str(completion.choices[0].message.content), prompt)


def term_query(term_1, term_2):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": 
          """
          You are a helpful assistant.
          """},
          {"role": "user", "content": "Describe the relationship or connection between %s and %s." %(term_1, term_2)}
      ]
    )
    return(str(completion.choices[0].message.content))

def prompt_correcter_query(messages):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages = messages
    )
    return(str(completion.choices[0].message.content))

def abbreviation_query(content, prompt="""
    Task Instructions:
    Develop an advanced natural language processing model capable of accurately associating contextual abbreviations or acronyms with their expanded forms. Employ the provided sample text to train the model effectively, aiming to produce output adhering to this format: {'Abbreviation': 'Expanded Form', ...}. The primary objective revolves around achieving precise identification and pairing of abbreviations alongside their respective expanded forms within the provided context.
    Desired Output:
    {'Abbreviation_1': 'Expanded_Form_1', 'Abbreviation_2': 'Expanded_Form_2'...}
    """):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": 
          """
          %s
          """ %prompt},
          {"role": "user", "content": "%s" %(content)}
      ]
    )

    return(str(completion.choices[0].message.content))

def score_return(response, expected):
    score = 0
    #true positives
    for key, value in expected.items():
        if key in response: 
            score += 1
            if response[key] == value:
                score += 1
        else:
            score -= 1
    #false positives
    for key, value in response.items():
        if key not in expected:
            score -= 1
    return(score)

def main():
    text_snippet = "./prompt_tests/abbreviation_text.txt"
    validation_snippet_file = "./prompt_tests/abbreviation_validation.txt"
    validation_response_file = "./prompt_tests/abbreviation_validation.tsv"
    chat_history = []

    with open(validation_snippet_file, 'r') as vfile:
        for line in vfile:
            validation_text = line.strip()
    validation_dict = {}
    df = pd.read_table(validation_response_file)
    for index, row in df.iterrows():
        validation_dict[row['expanded']] = row['abbreviation']
    #print(validation_text)
    #print(validation_dict)
    prompt_scores = {}
    with open(text_snippet, 'r') as tfile:
        for line in tfile:
            input_data = line.strip()

    df = pd.read_table("benchmark_abbreviations.tsv")
    return_dict = {}
    for index, row in df.iterrows():
        return_dict[row['expanded']] = row['abbreviation']

    i_scores = 3
    score_replicates = []
    for i in range(i_scores):
        answer = abbreviation_query(input_data, prompt)
        score = score_return(ast.literal_eval(answer), return_dict)
        score_replicates.append(score)
    avg_score = sum(score_replicates) / i_scores
    print(avg_score)
    print(score_replicates)
    print(prompt)
    sys.exit(0)
    all_messages = []
    for i in range(7):
        if i == 0:
            prompt = start_prompt
            message = {"role": "user", "content": "Can you help me update a ChatGPT prompt to give better results, given a prompt and a score where higher scores mean better results? I will iteratively provide a prompt and a score, and you give me back a modified/expanded/updated version of the prompt, while taking into account previous prompts and scoring history. Please return the updated prompt with no additional commentary, titles, headers, or footers."}
            all_messages.append(message)
            message = {"role": "system", "content": "Of course, I'd be happy to help! Please provide me with the prompt and its corresponding score, and I'll assist you in refining it."}
            all_messages.append(message)
            message = {"role":"user", "content": "Prompt: Create a model that can map abbreviations/acronyms to their expanded forms based on contextual information. Use the provided sample text to train the model to generate output in the format: {'Abbreviation': 'Expanded Form', ...}. The goal is to accurately identify and pair abbreviations with their corresponding expanded forms within the given context.\nScore: -8"}
            all_messages.append(message)
            prompt = prompt_correcter_query(all_messages)
            all_messages.append({"role":"system", "content":prompt})
        else:
            message = {"role":"user", "content": "The last prompt scored %s" %avg_score}
            all_messages.append(message)
        score_replicates = []
        i_scores = 3
        for i in range(i_scores):
            answer = abbreviation_query(input_data, prompt)
            score = score_return(ast.literal_eval(answer), return_dict)
            score_replicates.append(score)
        avg_score = sum(score_replicates) / i_scores
        all_messages.append({"role": "user", "content":"%s"%avg_score})
        print("iteration", i, "score", score)
        prompt_scores[i] = {"prompt":prompt, "answer":answer}
        prompt = prompt_correcter_query(all_messages)
        print(prompt)
        #sys.exit(0)
        

    with open("./prompt_tests/abb_pe.json", "w") as jfile:
        json.dump(prompt_scores, jfile)    

if __name__ == "__main__":
    main()
