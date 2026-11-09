import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

# OpenAI has no free embedding model. text-embedding-3-small is OpenAI's
# cheapest (~$0.02 per 1M tokens), called here through OpenRouter.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    # Send raw text, not tiktoken token IDs (OpenRouter expects text).
    check_embedding_ctx_length=False,
    # Must match the Pinecone index dimension (medium-blogs = 1024).
    dimensions=1024,
)

if __name__ == "__main__":
    print("Ingesting...")

    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    print(f"Uploading {len(texts)} chunks to Pinecone...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("Done")
