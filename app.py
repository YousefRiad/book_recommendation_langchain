# from flask import Flask, render_template, request
# import pandas as pd
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import DataFrameLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# import os

# app = Flask(__name__)

# # Load the dataset and prepare the model (similar to your book_recommendation.py)
# file_path = os.path.join('datasets', 'book_data', 'book_descriptions.csv')
# data_df = pd.read_csv(file_path)

# Embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
# loader = DataFrameLoader(data_df, page_content_column="description")
# descriptions = loader.load()
# chroma_db = Chroma.from_documents(descriptions, Embeddings)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     query = request.form['query']
#     relevant_docs = chroma_db.similarity_search(query, k=5)
#     results = [
#         {'title': doc.metadata['title'], 'description': doc.page_content}
#         for doc in relevant_docs
#     ] if relevant_docs else []
#     return render_template('index.html', results=results, query=query)

# if __name__ == '__main__':
#     app.run(debug=True)

# ****************************************************************************************

from flask import Flask, render_template, request
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os

app = Flask(__name__)

# Load the dataset and prepare the model (similar to your book_recommendation.py)
file_path = os.path.join('datasets', 'book_data', 'book_descriptions.csv')
data_df = pd.read_csv(file_path)

Embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
loader = DataFrameLoader(data_df, page_content_column="description")
descriptions = loader.load()
chroma_db = Chroma.from_documents(descriptions, Embeddings)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    k = min(int(request.form.get('k', 5)), 5)  # Get the 'k' value from the form and limit it to 5
    relevant_docs = chroma_db.similarity_search(query, k=k)
    results = [
        {'title': doc.metadata['title'], 'description': doc.page_content}
        for doc in relevant_docs
    ] if relevant_docs else []
    return render_template('index.html', results=results, query=query, k=k)

if __name__ == '__main__':
    app.run(debug=True)
