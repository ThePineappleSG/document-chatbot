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

from langchain.prompts.prompt import PromptTemplate

load_dotenv('.env')


print('Initialising...')


for i in trange(10):
    time.sleep(0.2)


vectordb = Chroma(persist_directory="./data", embedding_function=OpenAIEmbeddings())
vectordb.get()

print(' Vector store retrieved')


for i in trange(10):
    time.sleep(0.1)


print(' Cleared conversation history')


# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=True,
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []

for i in trange(10):
    time.sleep(0.15)

print(' Langchain doc_QA Initialised')

print(f"{yellow}---------------------------------------------------------------------------------")
print('                         Documents Loaded Please Ask Qns (WIP)                                ')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt (type q to exit): ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))