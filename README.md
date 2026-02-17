# 🌾 Agricultural RAG System

A complete Retrieval-Augmented Generation (RAG) system for agricultural knowledge with Gradio interface, LangServe API, and built-in evaluation metrics.

---

## 📋 Project Structure
agricultural-rag/
│
├── rag.py # Main application (single file)
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
└── evaluation_results.json # Generated after running evaluation

text

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📚 Knowledge Base** | 50+ agricultural documents covering crops, livestock, soil, irrigation, pests, and economics |
| **🔍 Semantic Search** | FAISS vector store with sentence-transformers embeddings |
| **🤖 Local LLM** | Google Flan-T5-small (no API keys required) |
| **💬 Chat Interface** | Gradio UI with source citations |
| **📊 Evaluation** | BLEU, METEOR, and relevance metrics with 15 test questions |
| **🌐 REST API** | LangServe endpoints for integration |
| **🚀 Single File** | Everything in one Python file for easy deployment |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | FastAPI + Uvicorn |
| **RAG Framework** | LangChain + LangServe |
| **Vector Database** | FAISS (CPU) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Language Model** | google/flan-t5-small |
| **UI Framework** | Gradio |
| **Evaluation** | NLTK (BLEU, METEOR) |

---

## 📦 Installation

### Step 1: Clone or create project directory
```bash
mkdir agricultural-rag
cd agricultural-rag
Step 2: Create the files
Create three files:

rag.py (copy the code from the provided source)

requirements.txt (copy the dependencies list)

README.md (copy this documentation)

Step 3: Install dependencies
bash
pip install -r requirements.txt
Step 4: Run the application
bash
python rag.py

⚙️ Configuration Options
Change the LLM Model
In rag.py, modify the initialization:

python
rag_system = AgriculturalRAGSystem(model_name="google/flan-t5-base")  # Larger model
Adjust Retriever Settings
python
self.retriever = self.vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diversity
    search_kwargs={"k": 5, "fetch_k": 10}
)
Add Your Own Documents
Extend the AGRICULTURAL_DOCS list:

python
AGRICULTURAL_DOCS.append("Your new agricultural text here...")
🚢 Deployment Options
Option 1: Local Server
bash
python rag.py
Option 2: Hugging Face Spaces
Create a new Space at huggingface.co/spaces

Choose "Gradio" as the SDK

Upload rag.py and requirements.txt

Set hardware to CPU (free tier)

Option 3: Background Process
bash
nohup python rag.py > output.log 2>&1 &