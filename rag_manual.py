from main import QUESTION, format_docs, llm, rag_prompt, retriever

# Step 1: search Pinecone for the 3 closest chunks.
docs = retriever.invoke(QUESTION)
print(f"Retrieved {len(docs)} chunks")

# Step 2: join them into one string for {context}.
context = format_docs(docs)

# Step 3: fill the prompt template.
prompt_value = rag_prompt.invoke({"context": context, "question": QUESTION})

# Step 4: send the filled prompt to the model.
response = llm.invoke(prompt_value.to_messages())
print(response.content)
