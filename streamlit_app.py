"""
Streamlit UI for War of the Worlds RAG System
"""

import streamlit as st
from rag_system import RAGSystem
import config

# Page configuration
st.set_page_config(
    page_title="War of the Worlds - RAG Q&A",
    page_icon="👽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
    }
    .source-box {
        background-color: #fff9e6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.chat_history = []

# Header
st.markdown('<h1 class="main-header">👽 The War of the Worlds</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about H.G. Wells\' classic novel using RAG</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"""
    **LLM:** Gemini Pro  
    **Embeddings:** Gemini Embeddings  
    **Vector DB:** ChromaDB  
    **Chunks:** {config.CHUNK_SIZE} chars  
    **Top K:** {config.TOP_K} results
    """)
    
    st.divider()
    
    if st.button("🔄 Reload Vector Database", use_container_width=True):
        st.session_state.rag_system = None
        st.session_state.initialized = False
        st.session_state.chat_history = []
        st.success("Ready to reload!")
    
    st.divider()
    
    st.header("📖 Sample Questions")
    sample_questions = [
        "Who is the narrator?",
        "What is the main plot of the story?",
        "How do the Martians attack Earth?",
        "What happens to the Martians in the end?",
        "Who is the artilleryman?",
        "What is the red weed?",
    ]
    
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.current_question = q

# Initialize RAG system
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing RAG System... This may take a minute..."):
        try:
            rag = RAGSystem(pdf_path="The_War_of_the_Worlds_NT.pdf", verbose=False)
            rag.initialize(force_reload=False)
            st.session_state.rag_system = rag
            st.session_state.initialized = True
            st.success("✅ RAG System initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing RAG system: {str(e)}")
            st.stop()

# Main content
st.divider()

# Question input
col1, col2 = st.columns([5, 1])

with col1:
    # Check if there's a sample question clicked
    default_question = st.session_state.get('current_question', '')
    question = st.text_input(
        "❓ Ask a question about The War of the Worlds:",
        value=default_question,
        placeholder="e.g., What is the main plot of the story?",
        key="question_input"
    )
    # Clear the current question after using it
    if 'current_question' in st.session_state:
        del st.session_state.current_question

with col2:
    ask_button = st.button("🔍 Ask", use_container_width=True, type="primary")

# Process question
if (ask_button or question) and question.strip():
    with st.spinner("🤔 Thinking..."):
        try:
            result = st.session_state.rag_system.query(question)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["source_documents"]
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display chat history (most recent first)
if st.session_state.chat_history:
    st.divider()
    st.header("💬 Conversation History")
    
    for idx, item in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {item["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br>{item["answer"]}</div>', unsafe_allow_html=True)
            
            # Show sources in expander
            with st.expander(f"📚 View {len(item['sources'])} Source Chunks"):
                for i, doc in enumerate(item["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                    <strong>Source {i} (Page {doc.metadata.get('page', 'N/A')}):</strong><br>
                    {doc.page_content[:300]}...
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
else:
    st.info("👆 Ask a question to get started! You can use the sample questions in the sidebar.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📚 RAG System Project - EEE 517 Deep Learning</p>
    <p>Book: <i>The War of the Worlds</i> by H.G. Wells</p>
</div>
""", unsafe_allow_html=True)

