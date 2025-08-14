import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import re
from datetime import datetime

# ML imports
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
import numpy as np

# Document processing
import PyPDF2

# Simple vector storage (no ChromaDB issues)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq                                # NEW
import os                                            # ensure os already imported


logger = logging.getLogger(__name__)

class SimpleLegalAnalyzer:
    """
    Simple, focused legal document analyzer
    Upload PDF -> Ask Questions -> Get Clear Answers
    """

    def __init__(self):
        # ----------  Groq API setup  ----------
        self.groq_client = None
        self.groq_model = "mixtral-8x7b-32768"
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("✓ Groq client initialized with Mixtral-8x7b-32768")
            except Exception as e:
                logger.warning(f"⚠️  Groq init failed: {e}")
        else:
            logger.warning("⚠️  GROQ_API_KEY not set – answers will not be enhanced")

        logger.info("Initializing Simple Legal Analyzer...")

        # Use smaller models to stay under 900MB
        self.device = "cpu"

        # Lightweight embedding model (22MB)
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # Lightweight QA model (240MB)
        logger.info("Loading QA model...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

        # Create QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model,
            tokenizer=self.qa_tokenizer,
            device=-1,  # CPU only
            max_length=512
        )

        # Simple in-memory storage
        self.documents = {}  # document_id -> {text, chunks, metadata}
        self.embeddings = {}  # document_id -> embeddings

        # Legal risk keywords for analysis
        self.risk_patterns = {
            "high_risk": [
                "unlimited liability", "waive rights", "no warranty", "irrevocable",
                "binding arbitration", "class action waiver", "no refund",
                "sell your data", "share with anyone", "change terms anytime"
            ],
            "medium_risk": [
                "may terminate", "collect information", "third parties", 
                "subject to change", "marketing purposes", "track your activity"
            ],
            "concerning_phrases": [
                "you agree that", "you acknowledge", "you waive", "you forfeit",
                "we may", "we reserve the right", "at our discretion"
            ]
        }

        logger.info("Simple Legal Analyzer ready!")

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                        continue

                if not text.strip():
                    return "Could not extract text from PDF. The document might be image-based or encrypted."

                return text.strip()

        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return f"Error reading PDF: {str(e)}"

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, exclamation, or question mark
                for punct in ['. ', '! ', '? ']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + 100:  # Ensure minimum chunk size
                        end = last_punct + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def analyze_legal_risks(self, text: str) -> Dict[str, Any]:
        """Analyze text for legal risks and concerning clauses"""
        try:
            # Handle empty or None text
            if not text or not text.strip():
                return self._create_empty_risk_analysis()
            
            text_lower = text.lower()

            # Find high risk patterns
            high_risks = []
            for pattern in self.risk_patterns.get("high_risk", []):
                if pattern in text_lower:
                    # Find the sentence containing this pattern
                    sentences = self._safe_split_sentences(text)
                    for sentence in sentences:
                        if sentence and pattern in sentence.lower():
                            high_risks.append({
                                "pattern": pattern,
                                "context": self._truncate_text(sentence.strip(), 200)
                            })
                            break

            # Find medium risk patterns
            medium_risks = []
            for pattern in self.risk_patterns.get("medium_risk", []):
                if pattern in text_lower:
                    sentences = self._safe_split_sentences(text)
                    for sentence in sentences:
                        if sentence and pattern in sentence.lower():
                            medium_risks.append({
                                "pattern": pattern,
                                "context": self._truncate_text(sentence.strip(), 200)
                            })
                            break

            # Find concerning phrases
            concerning = []
            for phrase in self.risk_patterns.get("concerning_phrases", []):
                if phrase in text_lower:
                    sentences = self._safe_split_sentences(text)
                    for sentence in sentences:
                        if sentence and phrase in sentence.lower():
                            concerning.append({
                                "phrase": phrase,
                                "context": self._truncate_text(sentence.strip(), 200)
                            })
                            break

            # Calculate risk score
            risk_score = len(high_risks) * 20 + len(medium_risks) * 10 + len(concerning) * 5
            risk_score = min(risk_score, 100)  # Cap at 100

            # Determine risk level
            if risk_score >= 60:
                risk_level = "HIGH"
            elif risk_score >= 30:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "high_risks": high_risks[:5],  # Limit to top 5
                "medium_risks": medium_risks[:5],
                "concerning_phrases": concerning[:5],
                "summary": self._generate_risk_summary(risk_level, risk_score, high_risks, medium_risks)
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_legal_risks: {e}")
            return self._create_error_risk_analysis(str(e))

    def _safe_split_sentences(self, text: str) -> List[str]:
        """Safely split text into sentences"""
        try:
            if not text:
                return []
            # Use a more robust sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"Error splitting sentences: {e}")
            return [text]  # Return original text as single sentence

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Safely truncate text to specified length"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _create_empty_risk_analysis(self) -> Dict[str, Any]:
        """Create a risk analysis for empty text"""
        return {
            "risk_score": 0,
            "risk_level": "LOW",
            "high_risks": [],
            "medium_risks": [],
            "concerning_phrases": [],
            "summary": "✅ LOW RISK (Score: 0/100)\nNo content available for analysis."
        }

    def _create_error_risk_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Create a risk analysis when there's an error"""
        return {
            "risk_score": 0,
            "risk_level": "UNKNOWN",
            "high_risks": [],
            "medium_risks": [],
            "concerning_phrases": [],
            "summary": f"⚠️ Analysis Error: {error_msg}\nCould not complete risk analysis."
        }

    def _generate_risk_summary(self, risk_level: str, score: int, high_risks: List, medium_risks: List) -> str:
        """Generate a human-readable risk summary"""
        try:
            if risk_level == "HIGH":
                summary = f"⚠️ HIGH RISK DOCUMENT (Score: {score}/100)\n"
                summary += "This document contains several concerning clauses that could significantly impact your rights. "
                if high_risks:
                    summary += f"Major concerns include {len(high_risks)} high-risk clauses that may limit your legal protections."
            elif risk_level == "MEDIUM":
                summary = f"⚡ MODERATE RISK (Score: {score}/100)\n"
                summary += "This document has some clauses that require attention. "
                if medium_risks:
                    summary += f"There are {len(medium_risks)} areas where the company retains broad discretion."
            else:
                summary = f"✅ LOW RISK (Score: {score}/100)\n"
                summary += "This document appears to be relatively user-friendly with standard terms."

            return summary
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return f"Risk Level: {risk_level} (Score: {score}/100)"

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """Add a PDF document for analysis"""
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Extract text from PDF
            logger.info(f"Extracting text from {file_path}...")
            text = self.extract_pdf_text(file_path)

            if text.startswith("Error") or text.startswith("Could not"):
                return {"error": text}

            # Generate document ID
            doc_id = hashlib.md5(text.encode()).hexdigest()[:8]

            # Check if already processed
            if doc_id in self.documents:
                return {"status": "already_exists", "document_id": doc_id}

            # Analyze document
            logger.info("Analyzing legal risks...")
            risk_analysis = self.analyze_legal_risks(text)

            # Create chunks
            logger.info("Creating text chunks...")
            chunks = self.chunk_text(text)

            # Generate embeddings for chunks
            logger.info("Generating embeddings...")
            try:
                embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                embeddings = np.zeros((len(chunks), 384))  # Default embedding size for all-MiniLM-L6-v2

            # Store document
            self.documents[doc_id] = {
                "text": text,
                "chunks": chunks,
                "file_path": file_path,
                "risk_analysis": risk_analysis,
                "created_at": datetime.now().isoformat()
            }

            self.embeddings[doc_id] = embeddings

            logger.info(f"Document {doc_id} processed successfully!")

            return {
                "status": "success",
                "document_id": doc_id,
                "risk_analysis": risk_analysis,
                "chunks_created": len(chunks),
                "filename": os.path.basename(file_path)
            }

        except Exception as e:
            logger.error(f"Error in add_document: {e}")
            return {"error": f"Processing error: {str(e)}"}

    def ask_question(self, question: str, document_id: str = None) -> Dict[str, Any]:
        """Ask a question about uploaded documents"""
        try:
            if not question.strip():
                return {"error": "Please provide a question"}

            if not self.documents:
                return {"error": "No documents uploaded. Please upload a PDF first."}

            # If no specific document, search all documents
            if document_id and document_id not in self.documents:
                return {"error": f"Document {document_id} not found"}

            # Search for relevant chunks
            relevant_chunks = []
            doc_ids = [document_id] if document_id else list(self.documents.keys())

            for doc_id in doc_ids:
                doc = self.documents[doc_id]
                doc_embeddings = self.embeddings[doc_id]

                # Generate question embedding
                question_embedding = self.embedding_model.encode([question], show_progress_bar=False)[0]

                # Calculate similarities
                similarities = cosine_similarity([question_embedding], doc_embeddings)[0]

                # Get top relevant chunks
                top_indices = similarities.argsort()[-3:][::-1]  # Top 3 chunks

                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Relevance threshold
                        relevant_chunks.append({
                            "text": doc["chunks"][idx],
                            "similarity": float(similarities[idx]),
                            "document_id": doc_id
                        })

            if not relevant_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question in the uploaded documents.",
                    "confidence": 0.0,
                    "method": "no_relevant_content"
                }

            # Sort by similarity and take top chunks
            relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            top_chunks = relevant_chunks[:3]  # Use top 3 most relevant

            # Combine context
            context = " ".join([chunk["text"] for chunk in top_chunks])[:1000]  # Limit context length

            # Use QA pipeline
            qa_result = self.qa_pipeline(
                question=question,
                context=context
            )

            # Generate comprehensive answer
            answer = self._generate_comprehensive_answer(
                question, qa_result["answer"], top_chunks, qa_result["score"]
            )

            return {
                "answer": answer,
                "confidence": float(qa_result["score"]),
                "method": "qa_with_context",
                "relevant_chunks": len(top_chunks),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {"error": f"Error processing question: {str(e)}"}

    def _generate_comprehensive_answer(self, question: str, basic_answer: str, chunks: List, confidence: float) -> str:
        """Generate a comprehensive, well-formatted answer using Groq API"""
        
        # Build basic context information
        confidence_text = ""
        if confidence > 0.8:
            confidence_text = "This information is clearly stated in the document."
        elif confidence > 0.5:
            confidence_text = "Based on the document content, this appears to be accurate."
        else:
            confidence_text = "The document mentions this, but the information may be incomplete."
        
        relevant_text = ""
        if chunks and len(chunks) > 0:
            relevant_text = chunks[0]['text'][:500]  # Get more context
        
        practical_advice = self._get_practical_advice(question)
        
        # ----------------------------------------------------------------
        # Use Groq API to create a well-formatted, natural response
        # ----------------------------------------------------------------
        if hasattr(self, 'groq_client') and self.groq_client:
            try:
                system_prompt = """You are a legal document analyst who explains complex legal terms in simple, clear English. Your job is to:

    1. Answer the user's question directly and clearly
    2. Explain what the legal language means in practical terms  
    3. Format your response professionally with proper paragraphs
    4. Use bullet points only when listing multiple items
    5. Write in a conversational, helpful tone
    6. DO NOT use markdown formatting like ** or * or \\n
    7. Write in plain text with proper paragraphs separated by line breaks

    Keep your response concise but comprehensive."""

                user_prompt = f"""The user asked: "{question}"

    Based on the legal document analysis:
    - Direct answer: {basic_answer}
    - Confidence level: {confidence_text}
    - Relevant document text: "{relevant_text}"
    - Practical guidance: {practical_advice}

    Please write a clear, well-formatted response that explains this in simple English. Do not use any markdown formatting symbols. Write in proper paragraphs with natural line breaks."""

                completion = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=600,
                    top_p=0.9
                )
                
                enhanced_answer = completion.choices[0].message.content.strip()
                
                # Clean up any remaining formatting issues
                enhanced_answer = enhanced_answer.replace('\\n', '\n')
                enhanced_answer = enhanced_answer.replace('\\*', '')
                enhanced_answer = enhanced_answer.replace('**', '')
                enhanced_answer = enhanced_answer.replace('*', '')
                
                return enhanced_answer
                
            except Exception as e:
                print(f"Groq enhancement failed: {e}")
                # Fall back to simple format
        
        # Fallback to clean, simple format if Groq not available
        response_parts = []
        response_parts.append(f"Answer: {basic_answer}")
        response_parts.append("")
        response_parts.append(f"Confidence: {confidence_text}")
        
        if relevant_text:
            response_parts.append("")
            response_parts.append("What the document says:")
            # Clean up the relevant text to be more readable
            clean_relevant = relevant_text.replace('\n', ' ').replace('  ', ' ').strip()
            if len(clean_relevant) > 200:
                clean_relevant = clean_relevant[:200] + "..."
            response_parts.append(f'"{clean_relevant}"')
        
        if practical_advice:
            response_parts.append("")
            response_parts.append(f"What this means for you: {practical_advice}")
        
        return "\n".join(response_parts)
    
    def _get_practical_advice(self, question: str) -> str:
        """Get practical advice based on question type"""
        try:
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['share', 'sell', 'data', 'information', 'privacy']):
                return "Pay attention to how your personal information is handled. Check if you can opt-out of data sharing."
            elif any(word in question_lower for word in ['terminate', 'cancel', 'end', 'close', 'delete']):
                return "Understand the conditions for account termination and what happens to your data afterward."
            elif any(word in question_lower for word in ['liability', 'responsible', 'damages', 'sue', 'legal']):
                return "This affects your legal rights and recourse options if something goes wrong."
            elif any(word in question_lower for word in ['change', 'modify', 'update', 'amend']):
                return "Check how you'll be notified of changes and whether you can reject them."
            elif any(word in question_lower for word in ['refund', 'money', 'payment', 'charge']):
                return "Understand the financial terms and your rights regarding payments and refunds."
            else:
                return "Review this clause carefully and consider how it might affect your rights and obligations."
        except Exception as e:
            logger.warning(f"Error generating practical advice: {e}")
            return "Please review this information carefully."

    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of a document"""
        try:
            if document_id not in self.documents:
                return {"error": "Document not found"}

            doc = self.documents[document_id]
            risk_analysis = doc["risk_analysis"]

            # Generate document type
            text_lower = doc["text"].lower()
            if "privacy policy" in text_lower:
                doc_type = "Privacy Policy"
            elif "terms of service" in text_lower or "terms of use" in text_lower:
                doc_type = "Terms of Service"
            elif "license" in text_lower:
                doc_type = "License Agreement"
            else:
                doc_type = "Legal Document"

            summary = {
                "document_type": doc_type,
                "filename": os.path.basename(doc["file_path"]),
                "risk_summary": risk_analysis.get("summary", "No summary available"),
                "risk_score": risk_analysis.get("risk_score", 0),
                "risk_level": risk_analysis.get("risk_level", "UNKNOWN"),
                "key_concerns": [],
                "created_at": doc["created_at"]
            }

            # Add key concerns
            high_risks = risk_analysis.get("high_risks", [])
            medium_risks = risk_analysis.get("medium_risks", [])
            
            if high_risks:
                summary["key_concerns"].extend([
                    f"🚨 HIGH RISK: {risk['pattern'].title()}" for risk in high_risks[:3]
                ])

            if medium_risks:
                summary["key_concerns"].extend([
                    f"⚠️ MEDIUM RISK: {risk['pattern'].title()}" for risk in medium_risks[:2]
                ])

            return summary

        except Exception as e:
            logger.error(f"Error in get_document_summary: {e}")
            return {"error": f"Error generating summary: {str(e)}"}

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents"""
        try:
            docs = []
            for doc_id, doc in self.documents.items():
                risk_analysis = doc.get("risk_analysis", {})
                docs.append({
                    "document_id": doc_id,
                    "filename": os.path.basename(doc["file_path"]),
                    "risk_level": risk_analysis.get("risk_level", "UNKNOWN"),
                    "risk_score": risk_analysis.get("risk_score", 0),
                    "created_at": doc["created_at"]
                })
            return docs
        except Exception as e:
            logger.error(f"Error in list_documents: {e}")
            return []

# Global analyzer instance
analyzer = SimpleLegalAnalyzer()

def create_legal_document_tool(document_path: str = None):
    """Legacy compatibility function"""
    global analyzer

    if document_path:
        result = analyzer.add_document(document_path)
        if result.get("status") != "success":
            logger.error(f"Failed to load document: {result}")

    def answer_function(question: str) -> str:
        result = analyzer.ask_question(question)
        return result.get("answer", "Unable to answer the question.")

    class CompatibilityTool:
        def __init__(self, func):
            self.func = func

    return CompatibilityTool(answer_function)

if __name__ == "__main__":
    # Test the analyzer
    print("Simple Legal Analyzer Test")
    print("=" * 40)

    # Example usage
    print("Analyzer initialized successfully!")
    print(f"Available documents: {len(analyzer.documents)}")