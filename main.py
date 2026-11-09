import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from ingestion import embeddings

load_dotenv()

QUESTION = "t What is Pinecone in machine learning?"

# LLM_MODEL is an OpenRouter preset (@preset/course-default), so OpenRouter picks the model.
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    # The prompt's {context} needs one string, but the retriever returns a list of Documents.
    return "\n\n".join(doc.page_content for doc in docs)


# RAG prompt: {context} gets the retrieved chunks, {question} gets the user's question.
rag_prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""
)


if __name__ == "__main__":
    docs = retriever.invoke(QUESTION)
    prompt_value = rag_prompt.invoke(
        {"context": format_docs(docs), "question": QUESTION}
    )
    print(prompt_value.to_string())
