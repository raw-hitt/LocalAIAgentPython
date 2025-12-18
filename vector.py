from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma #Vector store
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("pikachus_pizzeria_reviews_30.csv") #loading the data
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db" #database
add_documents = not os.path.exists(db_location) #checking if db exists

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Feedback"]+ " " + row["Branch"], #Data you need your model to lookup in the database
            metadata={"rating": row["Rating"], "date": row["Date"]}, #Additional information to use along with documents
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="pikachus_pizzeria_reviews", #collection name
    persist_directory=db_location, #db location
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids) #Adding documents in vector store
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} #Number of relevant documents model to check
)