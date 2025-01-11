import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from keybert import KeyBERT

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Initialize KeyBERT for Keyword Extraction
keybert_model = KeyBERT()

def extract_keywords(text_chunks):
    """
    Extract keywords from text chunks using KeyBERT and store them in metadata.
    """
    for chunk in text_chunks:
        keywords = keybert_model.extract_keywords(chunk.page_content, keyphrase_ngram_range=(1, 2), top_n=3)
        chunk.metadata["keywords"] = [kw[0] for kw in keywords]  # Store keywords in metadata
    return text_chunks

# Step 5: Extract Keywords
text_chunks_with_keywords = extract_keywords(text_chunks)

# Step 6: Store updated embeddings with keywords in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks_with_keywords, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"Vectorstore saved successfully at {DB_FAISS_PATH}")
