import os

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
conversation_retrieval_chain = None
chat_history = []

llm = None
llm_embeddings = None


# Initialize the llm and return the model with its embeddings
def init_llm():
    # Make sure to set the relevant api key in the environment if you are using a api based model like ones from openai (OPENAI_API_KEY)
    global llm, llm_embeddings
    llm = OpenAI()
    llm_embeddings = OpenAIEmbeddings()


def process_document(document_path):
    global conversation_retrieval_chain, llm, llm_embeddings

    loader = PyPDFLoader(document_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # create a vectorestore to use as the index
    db = Chroma.from_documents(texts, llm_embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # create a chain to answer questions
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)


def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    chat_history.append((prompt, result["answer"]))
    return result['answer']


init_llm()
