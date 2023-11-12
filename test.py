import os
import sys
from openai import OpenAI
from bardapi import Bard

"""
bard = Bard(token_from_browser=True)
answer = bard.get_answer("나와 내 동년배들이 좋아하는 뉴진스에 대해서 알려줘")['content']
print(answer)
sys.exit(0)
"""

client = OpenAI()
content="Bromocriptine mesylate is a semisynthetic ergot alkaloid derivative with potent dopaminergic activity. It inhibits prolactin secretion and may be used to treat dysfunctions associated with hyperprolactinemia. Bromocriptine is also indicated for the management of signs and symptoms of Parkinsonian Syndrome, as well as the treatment of acromegaly. Bromocriptine has been associated with pulmonary fibrosis, and can also cause sustained suppression of somatotropin (growth hormone) secretion in some patients with acromegaly. In 1995, the FDA withdrew the approval of bromocriptine mesylate for the prevention of physiological lactation after finding that bromocriptine was not shown to be safe for use.8,9 It continues to be used for the indications mentioned above."

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role": "system", "content": """You are a curator of scientific information, helping scientists understand drug mechanisms of action. Please format the following text into the schema, using 'Not Specified' to fill fields where information is uknown: 
       Drug:
       Disease: 
       DrugBank Accession Number: 
       Drug Target:
       Drug Cost: """},
    {"role": "user", "content": "%s" %content}
  ]
)

print(completion.choices[0].message)

content2 = "What is your response to this?"

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role": "system", "content": """You are a curator of scientific information, and your job is to make sure all information provided is correct, and to (1) fetch more documents that improve the accuracy of provided schema and (2) ask further questions about the returned information to deepen our understanding. (3) Please suggest additional fields that could be added the schema to best capture the information in a systemativ way. Please return your response in the following format:
       Updated Schema
       --------------

       Addtional Questions
       -------------------"""},
    {"role": "user", "content": "The original prompt was %s and the response was %s. %s" %(content, completion.choices[0].message, content2)}
  ]
)

print(completion.choices[0].message)

