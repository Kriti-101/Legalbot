import os
import re
import numpy as np
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from langchain.tools import Tool
from langchain.chains import LLMChain

class LegalDocumentQA:
    def __init__(self, document_path='sample_legal_document.pdf'):
        """
        Initialize the Legal Document QA system
        
        :param document_path: Path to the legal document
        """
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("SpaCy model not found. Please download with: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize Question Answering pipeline
        try:
            self.qa_pipeline = pipeline("question-answering")
        except Exception as e:
            print("Failed to load QA pipeline. Ensure transformers is installed.")
            raise
        
        # Initialize document content
        self.document_content = ""
        self.processed_sentences = []
        
        # Vectorization for semantic search
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Load document
        if not self.load_document(document_path):
            print("Failed to load the document. Please check the file path.")
    
    def load_document(self, document_path):
        """
        Load document from different file formats
        
        :param document_path: Path to the document
        :return: Boolean indicating successful loading
        """
        self.document_content = ""
        self.processed_sentences = []
        
        if not os.path.exists(document_path):
            print(f"Error: Document not found at {document_path}")
            return False
        
        _, ext = os.path.splitext(document_path)
        ext = ext.lower()
        
        try:
            if ext == '.pdf':
                with open(document_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            self.document_content += text + "\n"
            
            elif ext == '.docx':
                doc = docx.Document(document_path)
                self.document_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            elif ext == '.txt':
                with open(document_path, 'r', encoding='utf-8') as file:
                    self.document_content = file.read()
            
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            self._preprocess_document()
            return True
        
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def _preprocess_document(self):
        """
        Preprocess the document content
        Split into sentences and clean text
        """
        cleaned_text = re.sub(r'\s+', ' ', self.document_content).strip()
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        self.processed_sentences = [
            sent.strip() for sent in sentences if len(sent.strip().split()) > 3
        ]
    
    def find_context(self, question, top_k=3):
        """
        Find most relevant context for a question
        
        :param question: Input question
        :param top_k: Number of top contexts to return
        :return: List of most relevant contexts
        """
        if not self.processed_sentences:
            print("No document loaded or processed.")
            return []
        
        contexts = self.processed_sentences
        all_texts = [question] + contexts
        
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [contexts[idx] for idx in top_indices]
    
    def answer_question(self, question):
        """
        Answer a question based on the document
        
        :param question: Question to be answered
        :return: Detailed answer or explanation
        """
        if not self.processed_sentences:
            return "No document has been loaded. Please load a document first."
        
        contexts = self.find_context(question)
        
        if not contexts:
            return "Could not find relevant information to answer the question."
        
        best_answer = None
        best_score = 0
        
        for context in contexts:
            try:
                result = self.qa_pipeline(question=question, context=context)
                if result['score'] > best_score:
                    best_answer = result
                    best_score = result['score']
            
            except Exception as e:
                print(f"Error processing context: {e}")
        
        if best_answer and best_score > 0.1:
            return f"Answer: {best_answer['answer']} (Confidence: {best_answer['score']:.2f})"
        
        return "I could not find a definitive answer. Relevant contexts:\n" + \
               "\n".join(f"- {context}" for context in contexts)



# Wrap the LegalDocumentQA into a LangChain-compatible tool
def create_legal_document_tool(document_path):
    qa_system = LegalDocumentQA(document_path)
    
    def answer_with_context(question: str) -> str:
        return qa_system.answer_question(question)
    
    return Tool(
        name="LegalDocumentQA",
        func=answer_with_context,
        description="A QA tool for answering questions from legal documents."
    )


from fastapi import FastAPI
from pydantic import BaseModel
# Initialize FastAPI app
app = FastAPI()

# Path to your legal document
document_path = "sample_legal_document.pdf"  # Replace with your document path

# Create the LegalDocumentQA tool
qa_tool = create_legal_document_tool(document_path)

# Define input/output models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask-question/", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = qa_tool.func(request.question)
    return AnswerResponse(answer=answer)
# # Main execution
# def main():
#     # Initialize QA system with a specific document
#     qa_system = LegalDocumentQA('sample_legal_document.pdf')
    
#     # Continuous Q&A loop
#     print("Legal Document Q&A System")
#     print("Type 'quit' to exit")
    
#     while True:
#         # Get user question
#         question = input("\nAsk a question: ").strip()
        
#         # Check for exit
#         if question.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break
        
#         # Get and print answer
#         answer = qa_system.answer_question(question)
#         print(answer)


# if __name__ == "__main__":
#     main()