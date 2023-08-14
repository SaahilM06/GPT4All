from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# function for loading only TXT files
from langchain.document_loaders import TextLoader
# text splitter for create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# to be able to load the pdf files
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
# Vector Store Index to create our database about our knowledge
from langchain.indexes import VectorstoreIndexCreator
# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS
import os  #for interaaction with the files
import datetime
import sys
import json



## assign the path for the 2 models GPT4All and Alpaca for the embeddings 
gpt4all_path = './models/gpt4all-converted.bin' 
## REMOVED ## llama_path = './models/ggml-model-q4_0.bin' 
# Calback manager for handling the calls with  the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# create the embedding object
## REMOVED ## embeddings = LlamaCppEmbeddings(model_path=llama_path)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# create the GPT4All llm object
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

# Split text 
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=3) 
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources

# Load our local index vector db
index = FAISS.load_local("my_faiss_index", embeddings)

# create the prompt template
# Create the prompt template with context and question
template = """
Context: {context}
Question: {question}
Answer: """

# User-input question
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If command-line arguments are provided, use the first argument as the question
        question = sys.argv[1]
    else:
        # If no command-line argument is provided, read the question from stdin
        question = input("Your question: ")

    print("Your question:", question)


matched_docs, sources = similarity_search(question, index)

# Get the most relevant document's content to use as the context
context = sources[0]["page_content"]

# Update the prompt template to include both the context and the question
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context, question=question)

# Instantiate the LLMChain with the prompt dictionary
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Create a dictionary with both the context and question key-value pairs
input_dict = {
    "context": context,
    "question": question
}

# Run the LLMChain with the input dictionary
response = llm_chain.run(input_dict)

# Print the result
print("Response type:", type(response))
print("Response content:", response)

# Print the extracted content
print(response)
