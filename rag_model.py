import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import hashlib

# Lightweight ML imports
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# Document processing
import PyPDF2
import docx

# Lightweight vector storage
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class LightweightRAGModel:
    """
    Memory-efficient RAG model for legal document analysis
    Uses minimal RAM (<900MB) and no external APIs
    """

    def __init__(self, 
                 embedding_model="all-MiniLM-L6-v2",  # Only 22MB
                 qa_model="distilbert-base-cased-distilled-squad",  # Smaller than RoBERTa
                 max_chunk_size=300,  # Smaller chunks to save memory
                 device="cpu"):  # Force CPU to save GPU memory

        self.device = device
        self.max_chunk_size = max_chunk_size

        # Initialize lightweight embedding model (22MB)
        logger.info("Loading lightweight embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

        # Initialize small QA model (~250MB)
        logger.info("Loading lightweight QA model...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)

        # Create QA pipeline with memory optimization
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model,
            tokenizer=self.qa_tokenizer,
            device=-1,  # CPU only
            max_length=384,  # Shorter sequences
            doc_stride=128
        )

        # Initialize lightweight vector storage
        self.vector_store = self._init_vector_store()

        # Text splitter for chunking
        self.chunk_overlap = 50

        # Risk keywords for legal analysis (lightweight approach)
        self.risk_keywords = {
            "high_risk": [
                "irrevocable", "unlimited liability", "waive rights", "no warranty",
                "binding arbitration", "class action waiver", "data selling"
            ],
            "medium_risk": [
                "may terminate", "subject to change", "collect information", 
                "share data", "third parties", "marketing purposes"
            ],
            "low_risk": [
                "opt-out", "delete data", "user control", "limited liability",
                "reasonable notice", "privacy protection"
            ]
        }

        # Document cache (in-memory, limited size)
        self.document_cache = {}
        self.max_cache_size = 10  # Limit cache to save memory

        logger.info("Lightweight RAG model initialized successfully")

    def _init_vector_store(self):
        """Initialize lightweight vector storage"""
        if CHROMADB_AVAILABLE:
            try:
                # Use in-memory ChromaDB to save disk space
                client = chromadb.Client(Settings=ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=None,  # In-memory only
                    anonymized_telemetry=False
                ))

                collection = client.create_collection(
                    name="legal_docs",
                    metadata={"description": "Legal document chunks"}
                )

                logger.info("Initialized in-memory ChromaDB")
                return {"client": client, "collection": collection, "type": "chromadb"}

            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")

        # Fallback to simple in-memory storage
        logger.info("Using simple in-memory vector storage")
        return {"vectors": [], "texts": [], "metadata": [], "type": "simple"}

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks efficiently"""
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + 50:  # Ensure minimum chunk size
                    end = last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _calculate_document_hash(self, text: str) -> str:
        """Calculate hash for document deduplication"""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _assess_risk_score(self, text: str) -> Dict[str, Any]:
        """Lightweight risk assessment using keyword matching"""
        text_lower = text.lower()

        risk_counts = {"high_risk": 0, "medium_risk": 0, "low_risk": 0}
        found_keywords = {"high_risk": [], "medium_risk": [], "low_risk": []}

        for risk_level, keywords in self.risk_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    risk_counts[risk_level] += 1
                    found_keywords[risk_level].append(keyword)

        # Calculate overall score (0-100)
        total_score = (risk_counts["high_risk"] * 3 + 
                      risk_counts["medium_risk"] * 2 + 
                      risk_counts["low_risk"] * 1)

        normalized_score = min(total_score * 8, 100)  # Scale to 0-100

        risk_level = "high" if normalized_score > 60 else "medium" if normalized_score > 30 else "low"

        return {
            "overall_score": normalized_score,
            "risk_level": risk_level,
            "risk_breakdown": risk_counts,
            "found_keywords": found_keywords
        }

    def _detect_document_type(self, text: str) -> str:
        """Detect document type using keyword matching"""
        text_lower = text.lower()

        patterns = {
            "privacy_policy": ["privacy policy", "data protection", "personal information"],
            "terms_of_service": ["terms of service", "terms of use", "user agreement"],
            "eula": ["end user license", "software license", "eula"],
            "cookie_policy": ["cookie policy", "cookies", "tracking"]
        }

        max_matches = 0
        detected_type = "unknown"

        for doc_type, keywords in patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > max_matches:
                max_matches = matches
                detected_type = doc_type

        return detected_type

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add document to the lightweight knowledge base"""
        if not text.strip():
            return {"error": "Empty document"}

        # Check cache limit
        if len(self.document_cache) >= self.max_cache_size:
            # Remove oldest document
            oldest_key = next(iter(self.document_cache))
            del self.document_cache[oldest_key]

        # Calculate hash and check for duplicates
        doc_hash = self._calculate_document_hash(text)
        if doc_hash in self.document_cache:
            return {"status": "duplicate", "document_id": doc_hash}

        # Analyze document
        doc_type = self._detect_document_type(text)
        risk_analysis = self._assess_risk_score(text)

        # Create chunks
        chunks = self._chunk_text(text)

        doc_metadata = {
            "document_id": doc_hash,
            "document_type": doc_type,
            "risk_analysis": risk_analysis,
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }

        # Store in cache
        self.document_cache[doc_hash] = {
            "text": text,
            "chunks": chunks,
            "metadata": doc_metadata
        }

        # Add to vector store
        try:
            if self.vector_store["type"] == "chromadb":
                # Generate embeddings for chunks
                embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

                # Add to ChromaDB
                self.vector_store["collection"].add(
                    ids=[f"{doc_hash}_chunk_{i}" for i in range(len(chunks))],
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=[{**doc_metadata, "chunk_id": i} for i in range(len(chunks))]
                )
            else:
                # Simple vector storage
                embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    self.vector_store["vectors"].append(embedding)
                    self.vector_store["texts"].append(chunk)
                    self.vector_store["metadata"].append({**doc_metadata, "chunk_id": i})

        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")

        logger.info(f"Added document {doc_hash} with {len(chunks)} chunks")

        return {
            "status": "success",
            "document_id": doc_hash,
            "document_type": doc_type,
            "risk_score": risk_analysis["overall_score"],
            "chunks_created": len(chunks)
        }

    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        if not query.strip():
            return []

        try:
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

            if self.vector_store["type"] == "chromadb":
                # Query ChromaDB
                results = self.vector_store["collection"].query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(top_k, 10)
                )

                formatted_results = []
                if results["documents"] and results["documents"][0]:
                    for i in range(len(results["documents"][0])):
                        formatted_results.append({
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                        })

                return formatted_results

            else:
                # Simple vector search
                if not self.vector_store["vectors"]:
                    return []

                # Calculate similarities
                vectors = np.array(self.vector_store["vectors"])
                similarities = cosine_similarity([query_embedding], vectors)[0]

                # Get top results
                top_indices = similarities.argsort()[-top_k:][::-1]

                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Similarity threshold
                        results.append({
                            "text": self.vector_store["texts"][idx],
                            "metadata": self.vector_store["metadata"][idx],
                            "score": float(similarities[idx])
                        })

                return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def answer_question(self, question: str, context_text: str = None) -> Dict[str, Any]:
        """Answer question using RAG approach"""
        if not question.strip():
            return {"error": "Empty question"}

        contexts = []

        # If specific context provided, use it
        if context_text:
            contexts.append(context_text[:800])  # Limit context length

        # Search knowledge base for additional context
        search_results = self.search_documents(question, top_k=3)
        for result in search_results:
            if result["score"] > 0.4:  # Relevance threshold
                contexts.append(result["text"])

        if not contexts:
            return {
                "answer": "I don't have enough information to answer this question. Please provide a document or add more content to the knowledge base.",
                "confidence": 0.0,
                "method": "no_context"
            }

        # Combine contexts (limit total length for memory efficiency)
        combined_context = " ".join(contexts)[:1000]  # Max 1000 chars

        try:
            # Use QA pipeline
            qa_result = self.qa_pipeline(
                question=question,
                context=combined_context
            )

            # Calculate confidence based on model score and context relevance
            base_confidence = qa_result.get("score", 0.0)

            # Adjust confidence based on context quality
            if len(search_results) > 0:
                avg_search_score = sum(r["score"] for r in search_results) / len(search_results)
                adjusted_confidence = (base_confidence + avg_search_score) / 2
            else:
                adjusted_confidence = base_confidence * 0.8  # Lower confidence without search

            return {
                "answer": qa_result["answer"],
                "confidence": float(adjusted_confidence),
                "context_used": len(contexts),
                "method": "rag",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"QA pipeline error: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "method": "error"
            }

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Comprehensive document analysis"""
        if not text.strip():
            return {"error": "Empty document"}

        # Add to knowledge base (temporary analysis)
        result = self.add_document(text, {"analysis_only": True})

        if result.get("status") != "success":
            return result

        doc_id = result["document_id"]
        doc_info = self.document_cache.get(doc_id, {})

        analysis = {
            "document_type": result["document_type"],
            "risk_score": result["risk_score"],
            "character_count": len(text),
            "chunk_count": result["chunks_created"],
            "analysis_timestamp": datetime.now().isoformat()
        }

        # Add detailed risk analysis
        if "metadata" in doc_info:
            risk_info = doc_info["metadata"].get("risk_analysis", {})
            analysis["risk_breakdown"] = risk_info.get("risk_breakdown", {})
            analysis["found_keywords"] = risk_info.get("found_keywords", {})
            analysis["risk_level"] = risk_info.get("risk_level", "unknown")

        return analysis

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vector_count = 0
        if self.vector_store["type"] == "chromadb":
            try:
                vector_count = self.vector_store["collection"].count()
            except:
                pass
        else:
            vector_count = len(self.vector_store["vectors"])

        return {
            "cached_documents": len(self.document_cache),
            "vector_count": vector_count,
            "vector_store_type": self.vector_store["type"],
            "embedding_model": "all-MiniLM-L6-v2",
            "max_cache_size": self.max_cache_size,
            "device": self.device
        }

    def clear_cache(self):
        """Clear document cache to free memory"""
        self.document_cache.clear()
        logger.info("Document cache cleared")


# Legacy compatibility functions
def create_legal_document_tool(document_path: str = None):
    """Create tool compatible with existing ab_model.py interface"""

    rag_model = LightweightRAGModel()

    # Load document if provided
    if document_path and os.path.exists(document_path):
        try:
            # Extract text based on file type
            if document_path.endswith('.pdf'):
                with open(document_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            elif document_path.endswith('.txt'):
                with open(document_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            else:
                text = ""

            if text:
                rag_model.add_document(text, {"source": document_path})
                logger.info(f"Loaded document: {document_path}")

        except Exception as e:
            logger.error(f"Error loading document {document_path}: {e}")

    def answer_with_rag(question: str) -> str:
        """Answer function compatible with legacy interface"""
        result = rag_model.answer_question(question)

        if result.get("confidence", 0) > 0.1:
            return f"Answer: {result['answer']} (Confidence: {result['confidence']:.2f})"
        else:
            return result.get("answer", "Unable to find relevant information.")

    # Return object with func attribute for compatibility
    class LegacyTool:
        def __init__(self, func):
            self.func = func
            self.rag_model = rag_model

    return LegacyTool(answer_with_rag)


if __name__ == "__main__":
    # Test the lightweight RAG model
    print("Testing Lightweight RAG Model...")

    # Initialize model
    rag = LightweightRAGModel()

    # Test with sample legal text
    sample_text = """
    Privacy Policy

    We collect your personal information including name, email, and location data.
    This information may be shared with third parties for marketing purposes.
    We reserve the right to change this policy at any time without notice.
    You waive all rights to legal action against us.
    """

    # Add document
    print("\nAdding sample document...")
    result = rag.add_document(sample_text)
    print(f"Result: {result}")

    # Test question answering
    print("\nTesting question answering...")
    question = "Can they share my personal information?"
    answer = rag.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Test document analysis
    print("\nTesting document analysis...")
    analysis = rag.analyze_document(sample_text)
    print(f"Analysis: {analysis}")

    # Show stats
    print("\nSystem statistics:")
    stats = rag.get_stats()
    print(f"Stats: {stats}")
