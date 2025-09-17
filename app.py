from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize Flask app
app = Flask(__name__)

# --- Global Variables ---
CHROMA_PATH = "chroma_db"
rag_chain = None

def initialize_rag_chain():
    """
    Initializes the RAG chain by loading the vector store, LLM,
    and constructing the chain. This runs only once when the app starts.
    """
    global rag_chain

    # Load the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Load the existing vector store from disk
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construct the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain initialized successfully.")

# API endpoint to ask a question
@app.route("/ask", methods=["POST"])
def ask_question():
    if not rag_chain:
        return jsonify({"error": "RAG chain is not initialized"}), 500

    json_data = request.get_json()
    question = json_data.get("question")

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        response = rag_chain.invoke(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Initialize the RAG chain when the application starts
    initialize_rag_chain()
    # Run the Flask app
    app.run(debug=True, port=5001)