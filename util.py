import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')
apikey = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def try_openai_embeddings(text, model="text-embedding-3-small"):
   embed = client.embeddings.create(input=text, model=model)
   vectors = []
   for x in embed.data:
       vec = x.embedding
       vectors.append(vec)
   return vectors
