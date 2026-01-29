from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY from .env
client = OpenAI()

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input="hello world",
)

print(len(resp.data[0].embedding))
print(resp.data[0].embedding[:5])  # first 5 values