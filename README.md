# rag_pipeline
# 🚀 Advanced RAG Pipeline (Hybrid Retrieval + Reranking)

## 📌 Overview

This project implements an **Advanced Retrieval-Augmented Generation (RAG)** system for question answering over PDF documents.

It combines:

* 📄 Layout-aware PDF parsing
* 🧩 Hierarchical chunking (sections + abstracts + chunks)
* 🔍 Hybrid retrieval (Dense + Sparse)
* 🎯 Reranking for improved relevance
* 🤖 LLM-based answer generation

---

## 🏗️ Architecture

```
                ┌────────────────────┐
                │   User Uploads PDF │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   PDF Processing   │
                │ (Unstructured +    │
                │   pdfplumber)      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Document Creation │
                │  (Text + Tables +  │
                │   Metadata)        │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Hierarchical       │
                │ Chunking           │
                │ - Abstract         │
                │ - Chunks           │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Embedding Creation │
                │ (SentenceTransformers)
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Vector Store       │
                │ (FAISS Index)      │
                └─────────┬──────────┘
                          │
                          ▼
        ┌────────────────────────────────────┐
        │        Hybrid Retrieval            │
        │  ┌────────────┐   ┌────────────┐   │
        │  │ Dense (MMR)│   │ Sparse     │   │
        │  │ FAISS      │   │ BM25       │   │
        │  └─────┬──────┘   └─────┬──────┘   │
        │        └───────┬────────┘          │
        │                ▼                   │
        │      Ensemble Retriever           │
        └───────────────┬───────────────────┘
                        │
                        ▼
                ┌────────────────────┐
                │   Reranking        │
                │   (BM25Okapi)     │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Context Builder  │
                │ (Top Documents)    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │        LLM         │
                │ (GPT via OpenRouter)
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │     Final Answer   │
                └────────────────────┘
```

---

## ⚙️ Features

* ✅ Layout-aware PDF parsing using Unstructured
* ✅ Table extraction using pdfplumber
* ✅ Recursive chunking with overlap
* ✅ Custom embedding wrapper (SentenceTransformers)
* ✅ FAISS vector database
* ✅ MMR-based dense retrieval
* ✅ BM25-based sparse retrieval
* ✅ Ensemble hybrid retriever
* ✅ BM25 reranking
* ✅ Streamlit interactive UI

---

## 🧠 Key Concepts

* Retrieval-Augmented Generation (RAG)
* Hybrid Search (Dense + Sparse)
* Max Marginal Relevance (MMR)
* BM25 Ranking
* Vector Databases (FAISS)
* Prompt Engineering

---

## 🔄 Workflow

1. Upload PDF
2. Extract text + tables
3. Convert into structured documents
4. Perform hierarchical chunking
5. Generate embeddings
6. Store in FAISS
7. Retrieve using:

   * Dense (MMR)
   * Sparse (BM25)
8. Combine results (Hybrid Retrieval)
9. Rerank documents
10. Generate answer using LLM

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* SentenceTransformers
* FAISS
* BM25 (rank-bm25)
* Unstructured
* pdfplumber
* OpenRouter (LLM API)

---

## ▶️ Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```
OPEN_API_KEY=your_openrouter_api_key
```

---

## ▶️ Run the App

```bash
streamlit run rag_pipeline.py
```

---

## 💡 Example Use Cases

* 📄 Invoice analysis
* 📑 Research paper QA
* 🏢 Enterprise document search
* 📚 Knowledge base chatbot

---

## 🚀 Future Improvements

* Add cross-encoder reranking
* Multi-query retrieval
* Metadata filtering
* Vector DB (Pinecone / Weaviate)
* Streaming responses

---

## 👨‍💻 Author

Deepak Raj

* Python | Django | Angular | AI/LLM Engineer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
