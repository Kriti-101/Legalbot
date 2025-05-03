# Legalbot

This project implements a semantic question-answering (QA) system over legal documents (Indian Constitution, IPC, CPC) using LangChain, HuggingFace models, and vector databases.

## 📌 Overview

The system allows users to ask natural language questions about legal texts and receive accurate, contextual answers. It leverages text embedding, semantic search, and transformer-based models in a modular LangChain architecture.

---

## 🚀 Implementation

### 1. **Data Preparation**
- **Input:** PDF legal documents (Indian Constitution, IPC, CPC).
- **Extraction:** `PyPDF2` used to extract raw text.
- **Chunking:** Text is split using `RecursiveCharacterTextSplitter` to preserve context.
- **Embedding:** Sentence embeddings generated via `all-MiniLM-L6-v2` model from HuggingFace.

### 2. **Vector Store**
- **Tool:** `FAISS` for high-speed vector similarity search.
- **Storage:** Embeddings are stored to enable quick retrieval of relevant document chunks.

### 3. **Question Answering Pipeline**
- **Query Processing:** User query is converted into embedding.
- **Retrieval:** Relevant chunks retrieved from FAISS store.
- **Response Generation:** QA model (fine-tuned `deepset/roberta-base-squad2`) generates context-aware answers.

### 4. **LangChain Components**
- **LLMs:** Integrated using HuggingFace APIs.
- **Chains:** Combine prompt templates, memory, and model calls.
- **Agents:** Optional agents route complex queries through decision logic.

### 5. **LangSmith Integration**
- **Tracing:** Added using `@traceable` decorators.
- **Monitoring:** Real-time logging of inputs, outputs, and model behavior on LangSmith dashboard.
- **API Key Config:** Set using environment variables.

### 6. **Deployment**
- **Containerization:** Docker used with multi-stage builds to optimize image size.

---

## 🧰 Tools & Libraries

- **Frameworks:** LangChain, LangSmith
- **Embeddings & LLMs:** HuggingFace Transformers, all-MiniLM-L6-v2, SQuAD2 model
- **Libraries:** PyPDF2, FAISS, SpaCy, Transformers, ONNX
- **Infra:** Docker, Google Cloud Platform (GCP)
