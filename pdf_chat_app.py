import streamlit as st # this is for the interface
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embedding import OpenAIEmbeddings
from langchain.vectorstore import FAISS
from langchain.memory import ConversationBufferMemory
import os 
from dotenv import load_dotenv
import warnings
import tempfile
import openai

warnings.filterwarnings("ignore")

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Chat with your Mutual Fund Documents using LangChain and StreamLit")

uploaded_file = st.file_uploader("Upload a PDF", type = ["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix = ".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.NamedTemporaryFile
    
    st.success("PDF uploaded Successfully!")

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(model_name = "gpt-40"),
        retriever = vectorstore.as_retriever(),
        memory = memory
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your PDF:")

    if query:
        response = chain.run(question = query)
        st.session_state.chat_history.append(("You",query))
        st.session_state.chat_history.append(("AI",response))

    for speaker, text in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**{speaker}:** {text}")
        else:
            st.markdown(f"**{speaker}:** {text}")