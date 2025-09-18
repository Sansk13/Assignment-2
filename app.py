import streamlit as st
import os
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Global Variables ---
DATA_PATH = "Data/"
CHROMA_PATH = "chroma_db"

def create_vector_store():
    """
    Loads documents, splits them, creates embeddings, and
    persists them to a Chroma vector store.
    """
    documents = []
    # Load documents from the Data folder
    for f in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, f)
        if f.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
            documents.extend(loader.load())
        elif f.endswith('.pdf'):
            loader = PyPDFLoader(file_path=file_path)
            documents.extend(loader.load())
        elif f.endswith('.txt'):
            loader = TextLoader(file_path=file_path)
            documents.extend(loader.load())

    if not documents:
        st.error("No documents found in the Data folder.")
        return None

    st.write(f"Loaded {len(documents)} documents.")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_documents = text_splitter.split_documents(documents)
    st.write(f"Split documents into {len(split_documents)} chunks.")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Create the Chroma vector store from the documents and persist it
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    st.success("Vector store created successfully.")
    return vector_store

def initialize_rag_chain():
    """
    Initializes the RAG chain by loading the vector store, LLM,
    and constructing the chain.
    """
    # Create or load the vector store
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    retriever = vector_store.as_retriever()

    # Initialize the LLM pipeline
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question at the end.

Context: {context}
Question: {question}
""")

    # Construct the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit UI and Logic ---

st.title("Your RAG Chatbot")

# Button to create the vector store
if st.button("Create/Update Vector Store"):
    with st.spinner("Processing documents..."):
        create_vector_store()

# Initialize the RAG chain and session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = initialize_rag_chain()
    st.session_state.messages = []

# Display existing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and display the new message
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(prompt)
            st.markdown(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})