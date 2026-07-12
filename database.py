from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv  

load_dotenv()

data = PyPDFLoader(r"document loaders/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)
embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
      documents = chunks,
      embedding = embedding_model,
      persist_directory = "chroma_db"
)



