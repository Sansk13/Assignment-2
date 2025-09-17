import os
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Define the path for the source documents and the persistent vector store
DATA_PATH = "Data/"
CHROMA_PATH = "chroma_db"

def create_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and persists them to a Chroma vector store.
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
        print("No documents found in the Data folder.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_documents = text_splitter.split_documents(documents)
    print(f"Split documents into {len(split_documents)} chunks.")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Create the Chroma vector store from the documents and persist it
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print(f"Vector store created successfully and saved to {CHROMA_PATH}")


if __name__ == "__main__":
    create_vector_store()