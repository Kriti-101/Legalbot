from langsmith import wrappers, traceable
import logging
import os
# Assuming LegalDocumentQA is your class for handling QA
from ab_model import LegalDocumentQA  # Replace with the actual import path

# # Initialize the QA system (example setup)
# langsmith_api_key = "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"  # Replace with your LangSmith API key
# qa_system = LegalDocumentQA(langsmith_api_key=langsmith_api_key)

# Ensure LangSmith tracing is enabled

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="legalbot"

# Enable logging
logging.basicConfig(level=logging.INFO)

# Initialize the QA system (without langsmith_api_key)
qa_system = LegalDocumentQA()

@traceable  # Auto-trace this function
def pipeline(user_input: str):
    """
    A traced pipeline function to process user input using the LegalDocumentQA system.
    """
    # Load a sample document (if required for the pipeline)
    pdf_path = "sample_legal_document.pdf"  # Replace with your PDF path
    try:
        num_chunks = qa_system.load_document(pdf_path)
        logging.info(f"Document loaded with {num_chunks} chunks.")
    except Exception as e:
        logging.error(f"Failed to load document: {e}")
        return {"error": "Failed to load document"}

    # Use the QA system to answer the question
    try:
        result = qa_system.answer_question(user_input)
        return result
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return {"error": "Failed to process the question"}

# Example usage
response = pipeline("What is the role of shareholders?")
print("Response:", response)