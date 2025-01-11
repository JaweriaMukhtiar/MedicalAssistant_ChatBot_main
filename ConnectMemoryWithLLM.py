from dotenv import load_dotenv
import os
import time
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
from transformers import AutoModel, AutoTokenizer, pipeline

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate the token
if not HF_TOKEN:
    raise ValueError("Hugging Face Token is missing. Please set it in the environment.")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load BioBERT for embeddings
BIOBERT_MODEL = "dmis-lab/biobert-v1.1"
bio_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
bio_model = AutoModel.from_pretrained(BIOBERT_MODEL)

# Load GPT-2 for response generation
GPT2_MODEL = "gpt2"
gpt2_pipeline = pipeline("text-generation", model=GPT2_MODEL, tokenizer=GPT2_MODEL)

# Function to load the HuggingFace LLM
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Function to set the custom prompt for the QA chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Function to generate response with GPT-2
def generate_response_with_gpt2(prompt):
    gpt2_response = gpt2_pipeline(prompt, max_length=100, num_return_sequences=1)
    return gpt2_response[0]['generated_text']

# Function to safely invoke the QA chain with retry logic for rate limits
def safe_invoke(query, qa_chain):
    retries = 5
    for i in range(retries):
        try:
            response = qa_chain.invoke({'query': query})
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too many requests
                wait_time = 2 ** i  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise the error if it's not a rate limit issue
    raise Exception("Max retries reached. Could not complete the request.")

# Function to retrieve context using BioBERT embeddings and generate the answer
def bio_qa_chain(query):
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

        # Retrieve context using BioBERT embeddings
        context_docs = retriever.get_relevant_documents(query)
        context_text = "\n".join([doc.page_content for doc in context_docs])

        # Create GPT-2 prompt
        prompt = f"Context: {context_text}\nQuestion: {query}\nAnswer: "
        response = generate_response_with_gpt2(prompt)

        return response, context_docs

    except Exception as e:
        print(f"Error in QA chain: {str(e)}")
        return "Error occurred", []

# Load the database for FAISS and create the QA chain
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Interactive Query
user_query = input("Write Query Here: ")

# Option 1: Use the original QA chain with Hugging Face LLM
response = safe_invoke(user_query, qa_chain)
print("RESULT from Hugging Face LLM: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])

# Option 2: Use BioBERT and GPT-2 for QA
result, sources = bio_qa_chain(user_query)
print("RESULT from BioBERT + GPT-2: ", result)
print("SOURCE DOCUMENTS: ", sources)
