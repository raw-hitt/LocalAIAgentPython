from langchain_ollama.llms import OllamaLLM #Imports ollama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")  #specify the  model name, type ollama list in cmd

template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
""" #Template on what we want our model to behave based on the reviews data

prompt = ChatPromptTemplate.from_template(template) #Pass template in chat prompt
chain = prompt | model #will combine prompt , model 

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)