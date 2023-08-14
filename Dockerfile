FROM python:3.9


# Copy the required scripts and files to the /app directory inside the container
COPY readdocs.py /app/readdocs.py
COPY processquestion.py /app/processquestion.py
COPY index.faiss /app/index.faiss
COPY index.pkl /app/index.pkl
COPY gpt4all-converted.bin /app/models/gpt4all-converted.bin


# Create a directory to store the FAISS index
RUN mkdir /app/my_faiss_index


# Set the working directory to /app
WORKDIR /app


# Copy the documents folder from the host machine to the /app directory inside the container
COPY docs /app/docs


# Install the required dependencies
RUN pip install scikit-learn
RUN pip install pygpt4all==1.0.1
RUN pip install pyllamacpp==1.0.6
RUN pip install langchain==0.0.149
RUN pip install unstructured==0.6.5
RUN pip install pdf2image==1.16.3
RUN pip install pytesseract==0.3.10
RUN pip install pypdf==3.8.1
RUN pip install faiss-cpu==1.7.4
RUN pip install sentence_transformers


# Copy the requirements file to the /app directory inside the container
COPY requirements.txt .


# Run the readdocs.py script to process documents and create the FAISS index
# The output.csv file will be saved in the /app directory inside the container
CMD ["sh", "-c", "python readdocs.py docs my_faiss_index ./output.csv && python processquestion.py"]
