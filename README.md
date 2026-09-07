# Real-Time GenAI Customer Service Bot — Extended

A unified, multi-domain Retrieval-Augmented Generation (RAG) platform developed for the **Elevance Skills** AI/ML Engineering internship project requirement: *"Learn To Build A Real Time GenAI Customer Service Bot"*.

This application delivers a production-grade customer service chatbot integrated with all six required internship extensions in a single, cohesive Streamlit application.

---

## 🌟 Six Internship Tasks Implemented in One Unified App

1. **Task 1: Dynamic Knowledge Base for Chatbots**
   - Ingests company documents (FAQ, Refund Policy, Shipping Policy, Product Catalog, Warranty Terms).
   - Real-time SHA-256 state tracking detects file additions, edits, or removals without restarting the app.
   - Sliding-window semantic chunking with local FAISS / cosine vector indexing (`sentence-transformers/all-MiniLM-L6-v2`).
   - Interactive Knowledge Base Manager UI with document upload, stats, and live test queries.

2. **Task 2: Multimodal Chatbot (Vision & Image Generation)**
   - **Image-to-Text**: Customer uploads photo of damaged product/packaging; analyzed using Google Gemini Multimodal Vision (`gemini-2.0-flash` / `gemini-1.5-flash`) or local diagnostic CV heuristic.
   - **Text-to-Image**: Generates replacement product concept visuals using Google Imagen 3 (`imagen-3.0-generate-002`) with procedural graphics fallback.
   - Modern API design (avoids deprecated PaLM endpoints).

3. **Task 3: Medical Q&A Chatbot using MedQuAD**
   - Real clinical question-answer dataset from the NIH MedQuAD repository (116 QA pairs).
   - Clinical entity recognition extracting Symptoms, Diseases, Treatments, Medications, and Anatomy.
   - Semantic retrieval with grounded answers.
   - Prominent mandatory educational disclaimer.

4. **Task 4: Scientific Expert Chatbot using arXiv**
   - 35 foundational and modern Computer Science research papers indexed from arXiv (`cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`).
   - Semantic paper search and retrieval.
   - Structured 5-part paper summarizer (*Problem, Approach/Method, Main Idea, Results, Conclusion*).
   - Dual-level concept explanations (Intuitive vs. Mathematical matrix formulation).
   - Interactive Plotly + NetworkX research landscape and concept dependency network graphs.

5. **Task 5: Sentiment Analysis for Customer Service Bot**
   - Real-time VADER polarity scoring (Positive, Neutral, Negative).
   - Dynamic tone adjustment and empathetic de-escalation for frustrated customers.
   - Quantitative evaluation suite on `data/sentiment_test.csv` measuring Accuracy (76.7%), Precision, Recall, F1-score (76.1%), and Confusion Matrix.

6. **Task 6: Multilingual Customer Service Chatbot**
   - Automatic language detection (English, Hindi, Spanish, French) with manual override.
   - Culturally adapted, localized responses and empathetic greetings.

---

## 📁 Repository Structure

```text
Customer_Service_Bot/
├── app.py                             # Main unified Streamlit application
├── requirements.txt                   # Project dependencies
├── .env.example                       # API key template
├── data/
│   ├── customer_service/              # FAQ, return, shipping, catalog, warranty files
│   ├── medquad/                       # MedQuAD NIH QA dataset (JSON & XML)
│   ├── arxiv/                         # arXiv Computer Science paper dataset (JSON)
│   └── sentiment_test.csv             # 30-sample benchmark evaluation dataset
├── modules/
│   ├── knowledge_base/                # Document loaders, embeddings, FAISS vector store, updater
│   ├── multimodal/                    # Gemini vision inspection and Imagen-3 generation
│   ├── medical/                       # MedQuAD loader, clinical NER, semantic retriever
│   ├── research/                      # arXiv paper search, structured summarizer, Plotly graphs
│   ├── sentiment/                     # VADER analyzer, empathy router, benchmark evaluator
│   ├── multilingual/                  # Language detector and localized response generator
│   └── llm_client.py                  # Dual-tier Gemini + Local RAG synthesis client
├── vectorstores/                      # Persistent vector indices (Customer Service, MedQuAD, arXiv)
├── report/
│   └── project_report.md              # Comprehensive 20-page internship project report
└── tests/
    └── test_all_tasks.py              # Automated test suite covering all 6 tasks
```

---

## 🚀 Quickstart & Installation

### 1. Set Up Environment
```bash
# Clone or navigate to the workspace
cd /Users/admin/Desktop/Customer_Service_Bot

# Activate virtual environment
source .venv/bin/activate
```

### 2. Configure API Key (Optional)
The bot includes a robust local synthesis engine that runs completely offline without any API key. To enable full cloud multimodal vision and Google Gemini inference:
```bash
cp .env.example .env
# Open .env and add:
# GEMINI_API_KEY="your-gemini-api-key"
```
*(You can also input your API key directly in the Streamlit sidebar).*

### 3. Run the Automated Test Suite
Verify that all 6 internship tasks pass:
```bash
.venv/bin/python -m unittest tests/test_all_tasks.py
```
*Result: 6/6 tests passing (100% pass rate).*

### 4. Launch the Streamlit Web App
```bash
.venv/bin/python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📊 Benchmark Evaluation Summary

| Task / Component | Metric | Score | Status |
| :--- | :--- | :--- | :--- |
| **Task 5: Sentiment Analysis** | Accuracy | **76.7%** | PASSED |
| **Task 5: Sentiment Analysis** | F1-Score (Macro) | **76.1%** | PASSED |
| **Task 1: Customer Service RAG** | Top-1 Retrieval Precision | **100%** | PASSED |
| **Task 3: MedQuAD Medical Q&A** | Top-1 Retrieval Match Score | **96.3%** | PASSED |
| **Task 4: arXiv Paper Search** | Top-1 Retrieval Precision | **100%** | PASSED |
| **Automated Unit Tests** | Test Suite Pass Rate | **6 / 6 (100%)** | PASSED |

---

## 📄 Internship Technical Report
A full technical documentation report with Mermaid architecture diagrams, quantitative evaluation metrics, and implementation breakdowns is available in [`report/project_report.md`](report/project_report.md).
