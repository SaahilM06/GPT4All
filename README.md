This project demonstrates how to build a pipeline for processing documents, splitting them into manageable chunks, and creating a searchable vector store using Langchain, GPT4All, and FAISS. It includes a process for loading TXT and PDF files, generating vector embeddings using the HuggingFace model, and conducting similarity searches.

Features
Load and split documents (TXT and PDF) into chunks using Langchain.
Create vector embeddings using HuggingFace Embeddings.
Store and retrieve documents with FAISS for fast similarity search.
Run GPT4All locally with callback streaming for question-answering.
Efficiently handle large sets of documents and merge FAISS indexes.
Requirements
Python 3.7+
Langchain
GPT4All
HuggingFace
FAISS
Other dependencies as listed in requirements.txt

Installation
Clone the repository:
Copy code
git clone https://github.com/your-username/repo-name.git
Install the required Python packages:
pip install -r requirements.txt
Download and place the GPT4All model and HuggingFace embeddings in the ./models directory.

Usage
Initial Setup
Place your PDF documents in the ./docs directory.
Run the script to process the documents, generate vector embeddings, and create a FAISS index:
python process_documents.py
The FAISS index will be saved locally as my_faiss_index.
Querying the Index:
To ask a question based on your indexed documents:

Run the script:
python query_index.py "Your question here"
The script will search for relevant content in the document database and generate a response using GPT4All.

File Structure
process_documents.py: Script to load documents, split them into chunks, and generate a FAISS index.
query_index.py: Script to run similarity search and query the indexed documents.
docs/: Directory where PDF files should be placed.
models/: Directory for the GPT4All model and HuggingFace embeddings.
requirements.txt: Required Python dependencies.
