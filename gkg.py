import os
import streamlit as st
import mysql.connector
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("📄 Chat with your PDFs (Conversational RAG)")
st.write("Upload one or more PDF files and ask questions about their contents.")

# Groq API Key input
api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Unique session
    session_id = st.text_input("🆔 Session ID:", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []

        for uploaded_file in uploaded_files:
            temp_path = "./temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        chunks = text_splitter.split_documents(documents)

        # Store chunks in MySQL
        db_conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="root",
            database="langchain_db"
        )
        cursor = db_conn.cursor()

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                cursor.execute(
                    "INSERT INTO pdf_chunks (file_name, chunk_text, chunk_index) VALUES (%s, %s, %s)",
                    (file_name, chunk_text, idx)
                )

        db_conn.commit()
        cursor.close()
        db_conn.close()

        # FAISS vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        # Prompt to make question history-aware
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question that may reference context, rewrite it as a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Final QA system prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for answering questions using retrieved PDF context. If the answer is unknown, say 'I don't know'.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # RAG chain
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Session-based history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Ask question
        user_question = st.text_input("💬 Ask a question about the uploaded PDFs:")

        def is_question_related(question: str, retriever, threshold: float = 0.3) -> bool:
            docs = retriever.get_relevant_documents(question)
            return bool(docs)

        def get_bot_response(question: str) -> str:
            if not is_question_related(question, retriever):
                return "⚠️ Sorry, that question is unrelated to the provided PDFs."
            response = conversational_rag.invoke({"input": question}, config={
                "configurable": {"session_id": session_id}
            })
            return response["answer"]

        if user_question:
            response = get_bot_response(user_question)
            st.write("🤖 Assistant:", response)

else:
    st.warning("⚠️ Please enter your GROQ API key.")
