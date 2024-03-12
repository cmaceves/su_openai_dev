import sys
import ast
import json
import pandas as pd
from openai import OpenAI
client = OpenAI()

def extract_predicates(paragraph, entities, predicates):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           Extract semantic triplets from the given paragraph. Each triplet should consist of a subject, a predicate, and an object. A list of controlled vocabulary predicates will be provided and the predicate must be selected from the provided list. A list of entities that are likely to be subjects and objects will be provided, however triplets may include subjects and objects not given in the input. If no triplets are found, please return "no triplets". Do not return any additional commentary.

           Example Input:
           Paragraph: NSAIDs such as ibuprofen work by inhibiting the cyclooxygenase (COX) enzymes, which convert arachidonic acid to prostaglandin H2 (PGH2). PGH2, in turn, is converted by other enzymes to several other prostaglandins (which are mediators of pain, inflammation, and fever) and to thromboxane A2 (which stimulates platelet aggregation, leading to the formation of blood clots). Like aspirin and indomethacin, ibuprofen is a nonselective COX inhibitor, in that it inhibits two isoforms of cyclooxygenase, COX-1 and COX-2. Based on this mechanism, headaches are treated by ibuprofen. 
           Predicates:caused by
           symptoms of
           blocks
           treats
           negatively regulates
           stimulates
           Entities:NSAIDs
           ibuprofen
        
           Example Output:
           NSAIDs - negatively regulates - cyclooxygenase enzymes
           thromboxane A2 - stimulates - platelet aggregations
           ibuprofen - negatively regulates - COX-1
           ibuprofen - negatively regulates - COX-2
           aspirin - negatively regulates - COX-1
           aspirin - negatively regulates - COX-2
           indomethacin - negatively regulates - COX-1
           indomethacin - negatively regulates - COX-2
           ibuprofen - treats - headaches
           """},
          {"role": "user", "content": "Paragraph:%s\nEntities:%s\nPredicates:%s"%(paragraph, entities, predicates)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def test(paragraph, drug, disease):
    prompt =[
          {"role": "system", "content": 
           """
           Task:
           Given a paragraph of text, an input entitiy, and an output entity, please explain how the input entity is connected to the output entity. Do not return any additional commentary.

           Example Input
           Paragraph: Gastroesophageal reflux disease (GERD) or gastro-oesophageal reflux disease (GORD) is a chronic upper gastrointestinal disease in which stomach content persistently and regularly flows up into the esophagus, resulting in symptoms and/or complications. The most common symptoms of GERD in adults are an acidic taste in the mouth, regurgitation, and heartburn.[16] Less common symptoms include pain with swallowing/sore throat, increased salivation (also known as water brash), nausea,[17] chest pain, coughing, and globus sensation.[18] The acid reflux can induce asthma attack symptoms like shortness of breath, cough, and wheezing in those with underlying asthma.[18] Omeprazole is a selective and irreversible proton pump inhibitor. It suppresses stomach acid secretion by specific inhibition of the H+/K+-ATPase system found at the secretory surface of gastric parietal cells. Because this enzyme system is regarded as the acid (proton, or H+) pump within the gastric mucosa, omeprazole inhibits the final step of acid production.[50] Omeprazole also inhibits both basal and stimulated acid secretion irrespective of the stimulus[51] as it blocks the last step in acid secretion.[51] The drug binds non-competitively so it has a dose-dependent effect.[52] The inhibitory effect of omeprazole occurs within 1 hour after oral administration. The maximum effect occurs within 2 hours. The duration of inhibition is up to 72 hours. When omeprazole is stopped, baseline stomach acid secretory activity returns after 3 to 5 days. The inhibitory effect of omeprazole on acid secretion will plateau after 4 days of repeated daily dosing.[53]
           Starting:Omeprazole
           Ending:Gastroesophageal Reflux Disease 

           Example Output:
           Omeprazole supresses stomach acid secretion by inihibiting the H+/K+-ATPase system found on the secretary surface of gastic parietal cells. This inihibits the final step of acid production.
           """},
          {"role": "user", "content": "Paragraph:%s\nStarting:%s\nEnding:%s"%(paragraph, drug, disease)}
      ]


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)


def sentence_tense(sentence, pos):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": 
           """
           Task:
           You will be given a sentence, and the parts of speech for every word in that paragraph in the same order. Please rewrite the sentence so that no past tense or past participle verbs occur. Do not add any additional commentary or spacing.

           Example Input 1:
           Paragraph- John went to a crowded grocery store. He bought apples. He then ate the green apples.
           Parts of Speech- proper noun;verb past tense;infinite marker;determiner;verb past participle;noun;noun;personal pronoun;verb past tense;noun plural;personal pronoun;adverb;verb;determiner;adjective;noun plural;

           Example Output 1:
           John goes to a crowded grocery store. He buys apples. He eats the green apples.
           """},
          {"role": "user", "content": "Paragraph- %s\nParts of Speech- %s"%(sentence, pos)}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def triplets(text, entities):
    prompt=[
          {"role": "system", "content": 
           """
           Task:
           You will be given a paragraph, and a list of entities. Please extract all semantic triplets from the paragraph as they relate to the given entities. A semantic triplet consists of a subject, a predicate, and an object. The subject is the entity performing the action, the predicate is the action or relationship, and the object is the entity that undergoes the action. Either the object or the subject should be one of the given entities. Multiple triplets may appear in a single sentence. Seperate the subject, predicate, and object with "-" characters, and place each triplet on a newline. Do not return any additional commentary. If no triplets can be extracted, please return "no triplets".
           Example Input 1:
           Text: John goes to a crowded grocery store and buys an apple.
           Entities: ["apple"]

           Example Output 1:
           John - buys - apple
           """},
          {"role": "user", "content": "Text:%s\nEntities:%s"%(text, entities)}
      ]

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt
    )
    response = str(completion.choices[0].message.content)
    return(response, prompt)

def triplicates(text):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": 
           """
           Task:
           Given a sentence please extract all semantic triplets present. A semantic triplet consists of a subject, a predicate, and an object. The subject is the entity performing the action, the predicate is the action or relationship, and the object is the entity that undergoes the action. Multiple triplets may appear in a single sentence. Seperate the subject, predicate, and object with "-" characters, and place each triplet on a newline. Do not return any additional commentary. If no triplets can be extracted, please return "no triplets".
           Example Input 1:
           John goes to a crowded grocery store and buys an apple.

           Example Output 1:
           John - buys - apple
           John - goes to - grocery store
           """},
          {"role": "user", "content": "%s"%(text)}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)


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

def shorten_sentences(content):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": 
           """
           Task:
           Rewrite the following text such that every sentence contains no more than one subject, one object, and one verb. Create new sentences as needed, and make sure all nouns are preserved in the output. Use nouns in place of pronouns as often as possible. Do not return any additional commentary, formatting, or spacing.
           Example Input:
           The dog runs up the hill and barks at the cat. He wags his tail and greets the human ethusiastically. He sniffs the grass and the fire hydrant every morning.
           Example Output:
           The dog runs up the hill. The dog barks at the cat. The dog wags his tail. The dog greets the human enthusiastically. The dog sniffs the grass. The dog sniffs the fire hydrant.
           """},
        {"role": "user", "content": "%s"%content}
      ]
    )
    response = str(completion.choices[0].message.content)
    return(response)

def text_tense(content):
    """
    Change the tense to present and change present participles to present tense.
    """
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": 
            """
            Task
            ###
            Given a text, transform all present participle verb phrases into their present simple forms. Create new, shorter sentences as needed while keeping the original meaning of the text. The response may be repetitive, and do not return any additional commentary, formatting, spacing, or paragraph breaks. 

            Input Example
            ###
            The dog was running up the hill and over the river, jumping while he barked. I am beginning to see why everyone loves having pets. The dog plays his part in making me really happy by being joyful.
            
            Output Example
            ###
            The dogs runs up the hill and over the river. The dog jumps while he barks. I begin to see why everyone loves to have pets. The dog makes me really happy. The dog is joyful.
            """},
        {"role": "user", "content": "%s"%content}
      ]
    )

    response = str(completion.choices[0].message.content)
    return(response)

def replace_pronouns(content):
    """
    Format the text such that it contains as few prouns as possible.
    """
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
          {"role": "system", "content": 
           """
Given a body of text, completely replace all pronouns (e.g., he, she, it, they, etc.) with explicit nouns mentioned in the text. Additionally, substitute as many common nouns as possible with proper nouns that are explicitly stated within the provided content. Maintain each sentence as a standalone unit, without combining any sentences, even if it results in repetition. Focus solely on substituting pronouns with specific nouns from the text and replacing common nouns with appropriate proper nouns explicitly mentioned within the provided content. Ensure the tense of sentences remains unchanged during the substitution process.
          Do not return any additional commentary, paragraph spacing, or formatting.
           """},
        {"role": "user", "content": "%s"%content}
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
