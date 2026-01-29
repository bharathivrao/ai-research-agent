from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI()

resp = client.responses.create(
    model="gpt-5-mini",
    input="Hello! Explain what an AI agent does in one sentence."
)

print(resp.output_text)