# book_recommendation.py

# 1. Imports and Dependencies
import opendatasets as od
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
import re
import os

# Download the dataset
# od.download("https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset")

# Load and prepare the dataset
file_path = os.path.join('datasets', 'book_data', 'preprocessed_data.csv')

data_df = pd.read_csv(file_path)
data_df = data_df.dropna()
data_df = data_df.drop_duplicates()
data_df = data_df.sample(n=40000, random_state=42)
data_df = data_df.rename(columns={"book_title": "title", "Summary": "description"})

# 2. Model Initialization
Embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})

loader = DataFrameLoader(data_df, page_content_column="description")
descriptions = loader.load()

CHROMA_PATH = "chroma_books_db"

chroma_db = Chroma.from_documents(
    descriptions, Embeddings, persist_directory=CHROMA_PATH
)

# 3. Model Function
def get_book_recommendation(query):
    relevant_docs = chroma_db.similarity_search(query, k=5)
    if relevant_docs:
        return relevant_docs[0].metadata['title'], relevant_docs[0].page_content
    else:
        return None, "No relevant book found."

# 4. Example Usage
if __name__ == "__main__":
    # Example input description
    query = "Islam and the story of the prophet."
    
    # Call the function and print the result
    title, description = get_book_recommendation(query)
    print(f"Title: {title}\n")
    print(f"Description: {description}")
