from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Define paths
persist_directory = "./chroma_db"
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store the chain and retriever
_rag_chain = None
_retriever = None
_current_pdf = None

def setup_db(pdf_path, force_reload=False):
    """
    Set up the ChromaDB database. If force_reload is True or the database doesn't exist,
    load documents, create embeddings, and save to disk.
    """
    global _current_pdf
    
    # If database exists and we don't want to force reload, we can just load it
    if os.path.exists(persist_directory) and not force_reload and pdf_path == _current_pdf:
        print("Loading existing ChromaDB...")
        embedder = HuggingFaceEmbeddings()
        db = Chroma(persist_directory=persist_directory, embedding_function=embedder)
        return db
    
    print("Creating new ChromaDB...")
    # Load and split documents
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Create embeddings
    embedder = HuggingFaceEmbeddings()
    
    # Split documents for better semantic search
    text_splitter = SemanticChunker(embeddings=embedder)
    documents = text_splitter.split_documents(docs)
    
    # Create and persist the vector store
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=persist_directory
    )
    _current_pdf = pdf_path
    print(f"ChromaDB created with {len(documents)} chunks and saved to {persist_directory}")
    return db

def create_rag_chain(db):
    """Create the RAG chain using the provided database"""
    # Create retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Initialize LLM
    llm = OllamaLLM(model="llama3:8b")
    
    # Create prompt template
    prompt_template = """
    Answer the following question based on the provided context. If the answer cannot be found in the context,
    say "I cannot answer this based on the provided context."
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Format documents function
    def format_docs(docs):
        return "\n\n".join([f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}" for doc in docs])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def initialize_system(pdf_path, force_reload=False):
    """Initialize the RAG system and store the chain and retriever globally"""
    global _rag_chain, _retriever
    
    # Set up the database
    vector_db = setup_db(pdf_path, force_reload=force_reload)
    
    # Create and store the RAG chain and retriever
    _rag_chain, _retriever = create_rag_chain(vector_db)
    
    return {"status": "System initialized successfully"}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Save the file
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    
    try:
        # Initialize the system with the new PDF
        initialize_system(filename, force_reload=True)
        return jsonify({'message': 'File uploaded and processed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Check if the system is initialized
        if _rag_chain is None or _retriever is None:
            return jsonify({'error': 'System not initialized. Please upload a PDF first.'}), 400
        
        # Run the query through the RAG chain
        result = _rag_chain.invoke(data['query'])
        return jsonify({'response': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)