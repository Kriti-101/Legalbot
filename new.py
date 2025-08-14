from langsmith import wrappers, traceable
import logging
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel

from ab_model import LegalDocumentQA  # Replace with the actual import path

# Setup LangChain environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "legalbot"

logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app and QA system
app = FastAPI()
qa_system = LegalDocumentQA()

# Define request schema
class QuestionRequest(BaseModel):
    question: str

@traceable
def pipeline(user_input: str):
    pdf_path = "sample_legal_document.pdf"
    try:
        num_chunks = qa_system.load_document(pdf_path)
        logging.info(f"Document loaded with {num_chunks} chunks.")
    except Exception as e:
        logging.error(f"Failed to load document: {e}")
        return {"error": "Failed to load document"}

    try:
        result = qa_system.answer_question(user_input)
        return result
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return {"error": "Failed to process the question"}

# API route to accept questions
@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = pipeline(request.question)
    return {"response": response}
