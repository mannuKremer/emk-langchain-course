from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from main import QUESTION, format_docs, llm, rag_prompt, retriever

# The same 4 steps as rag_manual.py, connected with the | operator.
# Each step's output becomes the next step's input.
chain = (
    {
        "context": retriever | format_docs,  # question -> chunks -> one string
        "question": RunnablePassthrough(),  # question -> question (unchanged)
    }
    | rag_prompt  # {context, question} -> filled prompt
    | llm  # prompt -> model reply (AIMessage)
    | StrOutputParser()  # AIMessage -> plain string
)

print(chain.invoke(QUESTION))
