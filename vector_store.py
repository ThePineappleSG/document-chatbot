import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm, trange
import time
import shutil


load_dotenv('.env')

shutil.rmtree("./data")

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())


print('Initialising...')
# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)


for i in trange(10):
    time.sleep(0.2)

print(' Documents split into chunks')

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()

for i in trange(10):
    time.sleep(0.1)

print(' Embedded in Vector Store')


yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []

for i in trange(10):
    time.sleep(0.15)

print(' Documents Loaded')

print(f"{yellow}---------------------------------------------------------------------------------")
print('                         Documents Loaded Press Any Key to Exit                                ')
print('---------------------------------------------------------------------------------')

