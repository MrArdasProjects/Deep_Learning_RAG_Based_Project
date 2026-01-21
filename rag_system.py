"""
RAG System for "The War of the Worlds" by H.G. Wells
Using Google Gemini API and ChromaDB
"""

import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import config


class RAGSystem:
    def __init__(self, pdf_path: str, persist_directory: str = "./chroma_db", verbose: bool = True):
        """
        Initialize RAG System
        
        Args:
            pdf_path: Path to the PDF file
            persist_directory: Directory to persist ChromaDB
            verbose: Whether to print status messages
        """
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.verbose = verbose
        
        # Configure Gemini
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Initialize HuggingFace Embeddings (free and high-performance)
        if self.verbose:
            print("🔄 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        if self.verbose:
            print("✅ Embedding model loaded")
        
    def load_and_process_pdf(self):
        """Load PDF and split into chunks"""
        if self.verbose:
            print("📄 Loading PDF...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        if self.verbose:
            print(f"✅ Loaded {len(documents)} pages")
        
        # Split documents into chunks
        if self.verbose:
            print("✂️ Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        if self.verbose:
            print(f"✅ Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create ChromaDB vectorstore from chunks"""
        if self.verbose:
            print("🗄️ Creating vector store...")
            print("⏳ This may take a few minutes for the first run...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        if self.verbose:
            print(f"✅ Vector store created with {len(chunks)} embeddings")
        
    def load_vectorstore(self):
        """Load existing vectorstore"""
        if self.verbose:
            print("📂 Loading existing vector store...")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        if self.verbose:
            print("✅ Vector store loaded")
        
    def query(self, question: str):
        """Query the RAG system"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
        relevant_docs = retriever.invoke(question)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""You are an expert on the book "The War of the Worlds" by H.G. Wells.
Use the following pieces of context from the book to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always answer in English and provide detailed, accurate responses based on the book's content.

Context from the book:
{context}

Question: {question}

Detailed Answer:"""

        # Generate answer using Gemini
        generation_config = genai.types.GenerationConfig(
            temperature=config.TEMPERATURE
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return {
            "answer": response.text,
            "source_documents": relevant_docs
        }
    
    def initialize(self, force_reload: bool = False):
        """Initialize the entire RAG system"""
        
        # Check if vectorstore exists
        if os.path.exists(self.persist_directory) and not force_reload:
            if self.verbose:
                print("📂 Vector store already exists. Loading...")
            self.load_vectorstore()
        else:
            if self.verbose:
                print("🔄 Creating new vector store...")
            chunks = self.load_and_process_pdf()
            self.create_vectorstore(chunks)
        
        if self.verbose:
            print("\n🎉 RAG System initialized successfully!\n")


def main():
    """Main function for testing"""
    
    print("="*80)
    print("War of the Worlds - RAG System")
    print("="*80)
    print()
    
    # Initialize RAG system
    rag = RAGSystem(pdf_path="The_War_of_the_Worlds_NT.pdf")
    rag.initialize(force_reload=False)  # Set to True to recreate vectorstore
    
    # Test questions
    test_questions = [
        "Who is the main character in The War of the Worlds?",
        "What is the book about?",
        "How do the Martians die?",
    ]
    
    print("="*80)
    print("Testing RAG System with sample questions")
    print("="*80)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-"*80)
        
        try:
            result = rag.query(question)
            
            print(f"💡 Answer: {result['answer']}")
            print(f"\n📚 Sources: {len(result['source_documents'])} relevant chunks found")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("="*80)


if __name__ == "__main__":
    main()
