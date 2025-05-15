from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# Define paths
persist_directory = "./chroma_db"
pdf_path = "CVSS.pdf"

# Global variables to store the chain and retriever
_rag_chain = None
_retriever = None

def setup_db(force_reload=False):
    """
    Set up the ChromaDB database. If force_reload is True or the database doesn't exist,
    load documents, create embeddings, and save to disk.
    """
    # If database exists and we don't want to force reload, we can just load it
    if os.path.exists(persist_directory) and not force_reload:
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
    I will give you a snippet of code, using that and the pieces of context you're given, you are to 
    give the code a score from 1 to 10, depending on vulnerability where 1 is least vulnerable and 
    10 is most vulnerable.
    
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

def initialize_system(force_reload=False):
    """Initialize the RAG system and store the chain and retriever globally"""
    global _rag_chain, _retriever
    
    # Set up the database
    vector_db = setup_db(force_reload=force_reload)
    
    # Create and store the RAG chain and retriever
    _rag_chain, _retriever = create_rag_chain(vector_db)
    
    return {"status": "System initialized successfully"}

def query_answer(query):
    """Run a query and return the result"""
    global _rag_chain, _retriever
    
    # Check if the system is initialized
    if _rag_chain is None or _retriever is None:
        initialize_system()
    
    # Get the retrieved documents
    retrieved_docs = _retriever.invoke(query)
    
    # Run the query through the RAG chain
    result = _rag_chain.invoke(query)
    
    return {"result": result}

# Example usage
if __name__ == "__main__":
    # Initialize the system (only needs to be done once)
    initialize_system(force_reload=False)
    
    vulnerable_code = """
        #include <stdio.h>
        #include <string.h>
        #include <stdlib.h>

        void process_input(char *input) {
            char buffer[64];
            // Buffer overflow vulnerability
            strcpy(buffer, input);
            
            printf("Processing: %s\n", buffer);
            
            // Command injection vulnerability
            char cmd[100];
            sprintf(cmd, "echo %s", buffer);
            system(cmd);
            
            // Memory leak
            char *data = malloc(256);
            strcpy(data, "Sensitive information");
            // Missing free(data)
        }

        int main(int argc, char *argv[]) {
            if (argc < 2) {
                printf("Usage: %s <input>\n", argv[0]);
                return 1;
            }
            
            // No input validation
            process_input(argv[1]);
            
            return 0;
        }
    """
    
    # Just pass the query, no need for other parameters
    response = query_answer(vulnerable_code)
    print(response)