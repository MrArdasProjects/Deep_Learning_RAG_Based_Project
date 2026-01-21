# 👽 The War of the Worlds - RAG Q&A System

A Retrieval-Augmented Generation (RAG) system for answering questions about H.G. Wells' classic novel "The War of the Worlds" using LangChain, Google Gemini API, and ChromaDB.

## 🎯 Project Overview

This project is part of EEE 517 Deep Learning Methods and Applications course. It implements a RAG system that:
- Processes and chunks a 200-page PDF book
- Creates vector embeddings using Google Gemini Embeddings
- Stores embeddings in ChromaDB for efficient retrieval
- Answers questions using Gemini Pro LLM
- Provides an interactive Streamlit interface

## 🛠️ Technology Stack

- **LLM:** Google Gemini Pro
- **Embeddings:** Google Gemini Embeddings (models/embedding-001)
- **Vector Database:** ChromaDB
- **Framework:** LangChain
- **UI:** Streamlit
- **PDF Processing:** PyPDF

## 📦 Installation

1. Clone the repository or download the files

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure `The_War_of_the_Worlds_NT.pdf` is in the project directory

## 🚀 Usage

### Option 1: Streamlit Web Interface (Recommended)

Run the interactive web interface:
```bash
streamlit run streamlit_app.py
```

This will open a browser window where you can:
- Ask questions about the book
- View conversation history
- See source documents for each answer
- Use sample questions

### Option 2: Command Line

Run the RAG system directly:
```bash
python rag_system.py
```

This will:
1. Load and process the PDF
2. Create vector embeddings (first run only)
3. Run test questions
4. Display answers with sources

### Option 3: Use as a Library

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem(pdf_path="The_War_of_the_Worlds_NT.pdf")
rag.initialize()

# Ask questions
result = rag.query("What is the main plot of the story?")
print(result["answer"])
```

## ⚙️ Configuration

Edit `config.py` to customize:
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TEMPERATURE`: LLM temperature (default: 0.3)
- `TOP_K`: Number of relevant chunks to retrieve (default: 5)

## 📝 Sample Questions

- Who is the main character?
- What is the book about?
- How do the Martians attack Earth?
- What happens to the Martians in the end?
- Who is the artilleryman?
- What is the red weed?

## 📂 Project Structure

```
.
├── The_War_of_the_Worlds_NT.pdf  # Source book
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration settings
├── rag_system.py                  # Core RAG implementation
├── streamlit_app.py              # Web interface
├── chroma_db/                    # Vector database (auto-created)
└── README.md                     # This file
```

## 🔧 How It Works

1. **Document Loading:** PDF is loaded and split into pages
2. **Text Chunking:** Pages are split into ~1000 character chunks with overlap
3. **Embedding:** Each chunk is converted to a 768-dimensional vector using Gemini
4. **Vector Storage:** Embeddings are stored in ChromaDB for fast retrieval
5. **Query Processing:** User question is embedded and similar chunks are retrieved
6. **Answer Generation:** Gemini Pro generates an answer based on retrieved context

## 📊 Performance

- **Book Size:** ~200 pages
- **Total Chunks:** ~400-600 (depending on book length)
- **Embedding Dimension:** 768
- **Retrieval Time:** <1 second
- **Answer Generation:** 2-5 seconds

## 🎓 Course Information

- **Course:** EEE 517 Deep Learning Methods and Applications
- **Instructor:** Ayça Kumluca Topallı
- **Term:** Fall 2025
- **Due Date:** December 30, 2025

## 📚 References

- H.G. Wells - The War of the Worlds
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## ⚠️ Notes

- First run will take longer as it processes the PDF and creates embeddings
- Subsequent runs will be faster as it loads the existing vector database
- To force recreation of the vector database, set `force_reload=True` in initialization
- API rate limits: 15,000 requests/minute for Gemini API (free tier)

## 🤝 Contributing

This is a course project. Feel free to fork and modify for your own learning purposes!

## 📄 License

This project is for educational purposes only.


