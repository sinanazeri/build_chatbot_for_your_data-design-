import tempfile
import os

from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
conversation_retrieval_chain = None
chat_history = []


def process_document(document_data):
    global conversation_retrieval_chain
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(document_data)
        temp_file.flush()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        # select which embeddings we want to use
        embeddings = OpenAIEmbeddings()
        # create the vectorestore to use as the index
        db = Chroma.from_documents(texts, embeddings)
        # expose this index in a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        # create a chain to answer questions
        conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)


def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    chat_history.append((prompt, result["answer"]))
    return result['answer']
