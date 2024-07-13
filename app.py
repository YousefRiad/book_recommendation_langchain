from flask import Flask, render_template, request
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os

app = Flask(__name__)

# Load the dataset and prepare the model (similar to your book_recommendation.py)
file_path = os.path.join('datasets', 'book_data', 'preprocessed_data.csv')

data_df = pd.read_csv(file_path)

Embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
loader = DataFrameLoader(data_df, page_content_column="description")
descriptions = loader.load()
chroma_db = Chroma.from_documents(descriptions, Embeddings)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    relevant_docs = chroma_db.similarity_search(query, k=5)
    if relevant_docs:
        title = relevant_docs[0].metadata['title']
        description = relevant_docs[0].page_content
    else:
        title = "Not Found"
        description = "No relevant book found."

    return render_template('index.html', title=title, description=description, query=query)

if __name__ == '__main__':
    app.run(debug=True)
