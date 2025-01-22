import os

import dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


def choose_model():
    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI API...")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return llm, embeddings

    # Check for Gemini API key if OpenAI isn't available
    elif os.getenv("GEMINI_API_KEY"):
        print("Using Gemini API...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None, max_retries=2
        )
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        return llm, embeddings
    else:
        raise ValueError("No API key found for either OpenAI or Gemini.")


llm, embeddings = choose_model()

loader = TextLoader(file_path="data/return_policy.txt")
docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=20)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
doc_splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(documents=doc_splits, collection_name="rag_chroma", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

retrieve_return_policy = create_retriever_tool(
    retriever,
    "return_policy",
    "Search and return the relevant information about return policy",
)
