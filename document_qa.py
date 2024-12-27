import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

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
    
    def load_document(self, file_path):
        # Read PDF document
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        texts = text_splitter.split_text(text)
        
        # Create vector store for document retrieval
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        
        return len(texts)
    
    def answer_question(self, question, top_k=3):
        if not self.vectorstore:
            return "No document loaded. Please upload a document first."
        
        # Retrieve most relevant document chunks
        retrieved_docs = self.vectorstore.similarity_search(question, k=top_k)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        
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
        
        # Convert tokens back to text
        answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
        answer = self.tokenizer.decode(answer_tokens)
        
        return {
            'answer': answer,
            'context': context,
            'confidence': torch.max(start_logits).item()
        }


if __name__ == "__main__":
    print("Starting main execution...")
    # Instantiate the class
    qa_system = LegalDocumentQA()
    
    # Path to your PDF file
    pdf_path = "COI.pdf"  
    print("Loading document...")
    num_chunks = qa_system.load_document(pdf_path)
    print(f"Document loaded with {num_chunks} chunks.")
    
    # Ask a question
    question = "What are the laws?"  # Replace with your question
    print("Answering question...")
    result = qa_system.answer_question(question)
    
    # Print the result
    print("Answer:", result['answer'])
    print("Context:", result['context'])


