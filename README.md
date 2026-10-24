 🌾 Agricultural RAG System

A Retrieval-Augmented Generation (RAG) application that answers questions about agriculture using a curated agricultural knowledge base, semantic search, and a locally run language model. The project includes an interactive Gradio interface, a FastAPI/LangServe endpoint, and built-in evaluation utilities.

## ✨ Features

- **Agricultural Q&A:** Ask questions about crops, livestock, soil science, fertilizers, irrigation, pests, diseases, sustainable farming, precision agriculture, and agricultural economics.
- **Retrieval-Augmented Generation:** Retrieves relevant knowledge-base passages and uses them as context for answer generation.
- **Semantic search:** Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2` and indexes document chunks in a FAISS vector store.
- **Local language model:** Uses `google/flan-t5-small` through Hugging Face Transformers; no external LLM API key is configured in the code.
- **Source snippets:** Displays excerpts from retrieved passages alongside answers.
- **Interactive web UI:** Includes Chat, Evaluation, Knowledge Base, and About tabs in Gradio.
- **REST API:** Exposes the RAG chain through LangServe and provides health-check and evaluation endpoints through FastAPI.
- **Evaluation utilities:** Computes BLEU, METEOR, and a simple keyword-overlap relevance score against a set of reference answers.
- **Reproducible dependencies:** Dependency versions are pinned in `requirements.txt`.

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG orchestration | LangChain |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (`faiss-cpu`) |
| Text generation | Hugging Face Transformers |
| Language model | `google/flan-t5-small` |
| Web interface | Gradio |
| API framework | FastAPI |
| RAG API routes | LangServe |
| Evaluation | NLTK, NumPy, BLEU, METEOR |
| Model execution | CPU configuration in the current implementation |

## 🏗️ How It Works

```text
User question
     |
     v
Gradio Chat UI / API
     |
     v
FAISS Retriever <--- Query embedding
     |
     v
Top 4 relevant document chunks
     |
     v
Prompt + retrieved context
     |
     v
FLAN-T5-small (local generation)
     |
     v
Answer + retrieved source snippets
```

### RAG pipeline

1. **Knowledge preparation:** Agricultural knowledge entries are split into chunks using `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
2. **Embedding generation:** `sentence-transformers/all-MiniLM-L6-v2` converts chunks into vector representations.
3. **Indexing and retrieval:** FAISS stores the vectors; similarity search retrieves up to four relevant chunks per question.
4. **Context-grounded generation:** A prompt instructs the model to answer using the retrieved context and to say when the context does not contain the answer.
5. **Response display:** The Gradio chat interface displays the generated answer and snippets from retrieved documents.

## 📚 Knowledge Coverage

The built-in knowledge base includes information about:

- **Crops:** Maize/corn, wheat, rice, soybeans, and potatoes
- **Livestock:** Cattle, dairy farming, poultry, and swine
- **Soil science:** Soil texture, pH, organic matter, and NPK nutrients
- **Irrigation:** Drip, sprinkler, surface, deficit irrigation, and rainwater harvesting
- **Pests and diseases:** Common crop pests, plant diseases, and Integrated Pest Management (IPM)
- **Sustainable farming:** Organic farming, cover crops, crop rotation, conservation tillage, and agroforestry
- **Precision agriculture:** GPS-guided equipment, drones, NDVI, sensors, and variable-rate technology
- **Agricultural economics:** Commodity prices, futures markets, subsidies, and crop insurance

The project defines **15 evaluation questions** with corresponding reference answers.

## 🚀 Getting Started

### Prerequisites

- Python compatible with the pinned dependencies
- `pip`
- Internet access on first run to download the Hugging Face models and required NLTK resources
- Enough disk space and RAM for the embedding model and language model

> **Note:** The application runs inference on CPU in the current code. Initial model downloads and startup may take some time.

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python rag.py
```

When initialization finishes, open:

- **Web application:** http://localhost:8000
- **FastAPI / OpenAPI documentation:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

The application attempts to open the web interface automatically after startup. Stop the server with `Ctrl+C`.

## 🖥️ Application Tabs

| Tab | Purpose |
|---|---|
| **Chat** | Ask agricultural questions and review the answer with retrieved source snippets. |
| **Evaluation** | Run evaluation on the first five test questions and display sample results. |
| **Knowledge Base** | View the configured knowledge-base count, test-question count, model name, and sample entries. |
| **About** | Review a summary of the architecture, components, metrics, and local access URLs. |

## 🔌 API Reference

The FastAPI application exposes the following routes in the current implementation.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns a health status, configured model name, and number of agricultural knowledge entries. |
| `POST` | `/evaluate` | Runs evaluation across the full configured test set, saves results to `evaluation_results.json`, and returns the results. |
| `GET` | `/docs` | Opens the interactive FastAPI documentation. |
| LangServe routes | `/chain` | Exposes the LangChain RAG chain through LangServe. |

### Health-check example

```bash
curl http://localhost:8000/health
```

Example response shape:

```json
{
  "status": "healthy",
  "model": "google/flan-t5-small",
  "documents": 50
}
```

The `documents` value shown above is illustrative of the response format; the application calculates it from the knowledge-base list at runtime.

### Run the full evaluation through the API

```bash
curl -X POST http://localhost:8000/evaluate
```

The endpoint runs all configured evaluation questions and writes the results to `evaluation_results.json` in the application's working directory.

> **LangServe note:** The `/chain` route is generated by LangServe. Refer to the automatically generated API documentation for the exact request schema supported by the installed versions.

## 📊 Evaluation

The project compares generated answers with reference answers using three metrics:

| Metric | What it measures | Important limitation |
|---|---|---|
| **BLEU** | N-gram overlap between a reference answer and a generated answer. | Wording differences can lower the score even when an answer is useful. |
| **METEOR** | Token-level matching that can account for stemming and synonyms, depending on available resources. | It is still an automatic text-overlap metric, not a full factuality assessment. |
| **Answer Relevance** | A simple ratio based on overlap between non-stopword tokens in the question and answer. | It does not reliably measure semantic relevance or correctness. |

The evaluation dataset contains 15 questions and reference answers. The Gradio Evaluation tab evaluates the first five questions to reduce runtime; the `POST /evaluate` endpoint evaluates the full configured set.

**No benchmark scores are claimed in this README.** Run the evaluation in your environment to obtain scores for your installed dependencies and model setup.

## 📁 Project Structure

```text
.
├── rag.py              # Knowledge base, RAG pipeline, evaluation, API, and Gradio UI
├── requirements.txt    # Pinned Python dependencies
└── README.md           # Project documentation
```

`evaluation_results.json` is generated when the full evaluation endpoint runs successfully.

## ⚙️ Configuration

The current implementation defines these main settings in `rag.py`:

| Setting | Current value |
|---|---|
| Generation model | `google/flan-t5-small` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | Similarity search |
| Retrieved chunks (`k`) | `4` |
| Chunk size | `500` |
| Chunk overlap | `50` |
| Server host | `0.0.0.0` |
| Server port | `8000` |

These values can be adjusted in the source code to experiment with retrieval quality, latency, and resource usage.

## ⚠️ Limitations and Responsible Use

- **Domain coverage:** Answers are limited by the built-in knowledge base and the model's ability to use retrieved context.
- **Not a substitute for professional advice:** Agricultural recommendations can depend on local climate, soil tests, crop variety, growth stage, and regional regulations. Verify consequential decisions with a qualified agricultural specialist or trusted local guidance.
- **Evaluation is approximate:** BLEU, METEOR, and keyword overlap do not establish factual accuracy, completeness, or real-world agronomic suitability.
- **Local model resources:** The model and embedding components need to be downloaded and loaded at startup. Performance depends on available CPU, memory, and disk space.
- **Network exposure:** The server binds to `0.0.0.0`, which can make it reachable from other devices on the network. Do not expose it to an untrusted network or the public internet without appropriate authentication, access controls, and deployment hardening.
- **Dependency compatibility:** The project pins older library versions. Installation may require a compatible Python environment, and APIs may differ across library versions.

## 🛣️ Possible Future Improvements

- Load agricultural documents from external files (PDF, DOCX, CSV, or web resources) instead of relying only on hard-coded text.
- Add document metadata and clearer source attribution for each answer.
- Evaluate retrieval quality separately from answer-generation quality.
- Add semantic evaluation and human-reviewed factuality checks.
- Support configurable models, embedding models, and retrieval parameters.
- Add automated tests, structured logging, and clearer exception handling.
- Add authentication, rate limiting, and deployment configuration before hosting publicly.
- Provide localized guidance and region-specific agricultural references.

## 🤝 Contributing

Contributions and suggestions are welcome. A useful contribution should explain the problem being addressed, include reproducible steps, and update tests or documentation where appropriate.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Open a pull request describing the change.

---

**Built with Python, LangChain, FAISS, Hugging Face Transformers, FastAPI, and Gradio.**
