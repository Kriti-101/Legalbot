import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import logging
import langchain
from langchain_core.tracers.langchain import wait_for_all_tracers


import os
from langsmith import utils
from dotenv import load_dotenv

utils.tracing_is_enabled()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="legalbot"

load_dotenv(dotenv_path=".env",override=True)



# Set up LangChain Tracing Configuration
langsmith_api_key = "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" # Replace with your LangSmith API Key
project_id = "legalbot" # Replace with your LangSmith project ID

# Optional: If you want to use a custom logger, you can also set up your logging here
logger = logging.getLogger("LegalDocumentQA")
logging.basicConfig(level=logging.INFO)

class LegalDocumentQA:
    def __init__(self, 
                 model_name="deepset/roberta-base-squad2", 
                 embedding_model="all-MiniLM-L6-v2"):
        # Load QA model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Load embedding model for document retrieval
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Vector store for document chunks
        self.vectorstore = None

    def log_event(self, event_name, details=None):
        """
        Custom method to log events. Also integrates with LangChain's tracing system.
        :param event_name: Name of the event.
        :param details: Additional details about the event (optional).
        """
        if details:
            logger.info(f"Event: {event_name}, Details: {details}")
        else:
            logger.info(f"Event: {event_name}")

    def load_document(self, file_path):
        """
        Load a PDF document, split it into chunks, and store it in a vector store.
        :param file_path: Path to the PDF file.
        :return: Number of chunks created.
        """
        try:
            # Read PDF document
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    self.log_event("EmptyPageWarning", {"page_number": reader.pages.index(page)})

            if not text.strip():
                raise ValueError("The document contains no extractable text.")

            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100
            )
            texts = text_splitter.split_text(text)

            # Create vector store for document retrieval
            self.vectorstore = FAISS.from_texts(texts, self.embeddings)

            # Log document processing event
            self.log_event("load_document", {"file_path": file_path, "num_chunks": len(texts)})

            return len(texts)

        except Exception as e:
            self.log_event("DocumentLoadError", {"file_path": file_path, "error": str(e)})
            raise

    def answer_question(self, question, top_k=3):
        """
        Answer a question based on the loaded document.
        :param question: The question to answer.
        :param top_k: Number of top relevant document chunks to retrieve.
        :return: Dictionary containing the answer, context, and confidence score.
        """
        if not self.vectorstore:
            return "No document loaded. Please upload a document first."

        try:
            # Retrieve most relevant document chunks
            retrieved_docs = self.vectorstore.similarity_search(question, k=top_k)
            context = " ".join([doc.page_content for doc in retrieved_docs])

            if not context.strip():
                return {
                    'answer': None,
                    'context': None,
                    'confidence': 0.0,
                    'error': "No relevant context found for the question."
                }

            # Prepare inputs for QA model
            inputs = self.tokenizer(
                question, 
                context, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )

            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract answer
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            start_index = torch.argmax(start_logits)
            end_index = torch.argmax(end_logits)

            if start_index > end_index:
                return {
                    'answer': None,
                    'context': context,
                    'confidence': 0.0,
                    'error': "Model produced an invalid token span."
                }

            # Convert tokens back to text
            answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Log the query and response
            self.log_event("answer_question", {
                "question": question,
                "answer": answer,
                "context": context,
                "confidence": torch.max(start_logits).item()
            })

            return {
                'answer': answer,
                'context': context,
                'confidence': torch.max(start_logits).item()
            }

        except Exception as e:
            self.log_event("AnswerQuestionError", {"question": question, "error": str(e)})
            return {
                'answer': None,
                'context': None,
                'confidence': 0.0,
                'error': str(e)
            }

# Now create an instance of LegalDocumentQA and execute it
print("Starting main execution...")

qa_system = LegalDocumentQA()

# Path to your PDF file
pdf_path = "sample_legal_document.pdf"  
print("Loading document...")
num_chunks = qa_system.load_document(pdf_path)
print(f"Document loaded with {num_chunks} chunks.")

# Ask a question
question = "what is corporate law?"  # Replace with your question
print("Answering question...")
result = qa_system.answer_question(question)

# Print the result
print("Answer:", result['answer'])
print("Context:", result['context'])

# Wait for all tracing events to finish
wait_for_all_tracers()
