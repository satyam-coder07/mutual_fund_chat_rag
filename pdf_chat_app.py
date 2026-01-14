import streamlit as st
import os
import tempfile
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Chat with your Mutual Fund Documents using LangChain and StreamLit")

uploaded_file = st.file_uploader("Upload a PDF", type = ["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix = ".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.success("PDF uploaded Successfully!")

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")),
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