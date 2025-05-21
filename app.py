import os
from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableMap
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests



## Enabling Langsmith Tracking
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

# Loading data for web scrapping 
loader=WebBaseLoader("https://docs.smith.langchain.com/administration/tutorials/manage_spend")
docs=loader.load()

#Convering into text
textSplitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=textSplitter.split_documents(docs)

#Applying embeddings to the converted texts
embeddings=(
    OllamaEmbeddings(model="gemma:2b") #by default it uses lama2
)

#Creating and storing vector data 
vectorstoredb=FAISS.from_documents(documents,embeddings)

#To get a callable obebject to map result
retriever=vectorstoredb.as_retriever()
# This gets relevant context documents and prepares inputs for the prompt
retrieval_chain = (
    RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: "\n\n".join(
            doc.page_content for doc in retriever.get_relevant_documents(x["question"])
        )
    })
)

# Now update the prompt to use context and question
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the context to answer the question."),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# Model
llm=OllamaLLM(model="llama3")

# For parsing output
output_parser=StrOutputParser()

# Final chain
chain = retrieval_chain | prompt | llm | output_parser

## Designing streamlit framework
st.title("Langchain Demo with LLAMA3")
input_text=st.text_input("What question you have in mind")

if input_text:
    st.write(chain.invoke({"question":input_text}))
    