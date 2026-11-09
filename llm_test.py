from main import QUESTION, llm

# No retrieved context: this shows what the model knows on its own.
response = llm.invoke(QUESTION)
print(response.content)
