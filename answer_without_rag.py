import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM_MODEL is an OpenRouter preset (@preset/course-default), so OpenRouter picks the model.
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

question = "What is Pinecone in machine learning?"

# No retrieved context: this shows what the model knows on its own.
response = llm.invoke(question)
print(response.content)
