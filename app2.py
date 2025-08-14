from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import our simple analyzer
from analyser import SimpleLegalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Simple Legal Document Analyzer",
    description="Upload PDFs and get easy-to-understand legal analysis",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer (global instance)
analyzer = SimpleLegalAnalyzer()

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    method: str
    relevant_chunks: int

# Routes
@app.get("/", response_class=HTMLResponse)
def main_page():
    """Simple, clean interface for legal document analysis"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal Document Analyzer</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }

            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }

            .content {
                padding: 30px;
            }

            .section {
                margin-bottom: 30px;
                padding: 25px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: #fafafa;
            }

            .section h2 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.4em;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .file-upload {
                border: 2px dashed #3498db;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                background: white;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .file-upload:hover {
                background: #f8f9fa;
                border-color: #2980b9;
            }

            .file-upload input[type="file"] {
                display: none;
            }

            .upload-text {
                font-size: 1.1em;
                color: #555;
                margin-bottom: 10px;
            }

            .upload-subtitle {
                color: #888;
                font-size: 0.9em;
            }

            input[type="text"], textarea {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 1em;
                margin: 10px 0;
                font-family: inherit;
            }

            textarea {
                min-height: 100px;
                resize: vertical;
            }

            button {
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 6px;
                font-size: 1em;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 5px;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
            }

            button:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 8px;
                background: white;
                border-left: 4px solid #3498db;
                display: none;
            }

            .result.show {
                display: block;
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .risk-high { color: #e74c3c; font-weight: bold; }
            .risk-medium { color: #f39c12; font-weight: bold; }
            .risk-low { color: #27ae60; font-weight: bold; }

            .docs-list {
                background: white;
                border-radius: 6px;
                padding: 15px;
                margin-top: 15px;
            }

            .doc-item {
                padding: 10px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .doc-item:last-child {
                border-bottom: none;
            }

            .status {
                padding: 15px;
                margin: 10px 0;
                border-radius: 6px;
                font-weight: 500;
            }

            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }

            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }

            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-left: 10px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 Legal Document Analyzer</h1>
                <p>Upload any legal document and get clear, understandable analysis</p>
            </div>

            <div class="content">
                <!-- Upload Section -->
                <div class="section">
                    <h2>📤 Upload Legal Document</h2>
                    <div class="file-upload" onclick="document.getElementById('fileInput').click()">
                        <input type="file" id="fileInput" accept=".pdf" onchange="uploadFile()">
                        <div class="upload-text">
                            <strong>Click to upload PDF document</strong>
                        </div>
                        <div class="upload-subtitle">
                            Privacy Policies, Terms of Service, Contracts, etc.
                        </div>
                    </div>
                    <div id="uploadStatus"></div>
                </div>

                <!-- Documents List -->
                <div class="section">
                    <h2>📚 Uploaded Documents</h2>
                    <div id="documentsList">
                        <div style="text-align: center; color: #888; padding: 20px;">
                            No documents uploaded yet
                        </div>
                    </div>
                    <button onclick="refreshDocuments()">Refresh List</button>
                </div>

                <!-- Q&A Section -->
                <div class="section">
                    <h2>❓ Ask Questions</h2>
                    <input type="text" id="questionInput" placeholder="e.g., 'Can they share my personal data?' or 'What happens if I cancel?'">
                    <button id="askBtn" onclick="askQuestion()">Ask Question</button>

                    <div id="answerResult" class="result"></div>

                    <div style="margin-top: 20px;">
                        <h3>💡 Example Questions:</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                            <button onclick="askExample('Can they share my personal information?')">Data Sharing</button>
                            <button onclick="askExample('What happens if I want to cancel?')">Cancellation</button>
                            <button onclick="askExample('Am I liable for damages?')">Liability</button>
                            <button onclick="askExample('Can they change the terms?')">Terms Changes</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentDocuments = [];

            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) return;

                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    showStatus('Please upload a PDF file only.', 'error');
                    return;
                }

                const statusDiv = document.getElementById('uploadStatus');
                statusDiv.innerHTML = '<div class="status">Uploading and analyzing document... <div class="loading"></div></div>';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/upload/', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        const riskClass = result.risk_analysis.risk_level.toLowerCase();
                        statusDiv.innerHTML = `
                            <div class="status success">
                                <strong>✅ Document uploaded successfully!</strong><br>
                                File: ${result.filename}<br>
                                Risk Level: <span class="risk-${riskClass}">${result.risk_analysis.risk_level}</span><br>
                                Risk Score: ${result.risk_analysis.risk_score}/100<br>
                                ${result.risk_analysis.summary}
                            </div>
                        `;
                        refreshDocuments();
                    } else {
                        statusDiv.innerHTML = `<div class="status error">❌ Error: ${result.detail || 'Upload failed'}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="status error">❌ Network error: ${error.message}</div>`;
                }

                // Reset file input
                fileInput.value = '';
            }

            async function refreshDocuments() {
                try {
                    const response = await fetch('/documents/');
                    const documents = await response.json();

                    const listDiv = document.getElementById('documentsList');

                    if (documents.length === 0) {
                        listDiv.innerHTML = '<div style="text-align: center; color: #888; padding: 20px;">No documents uploaded yet</div>';
                        return;
                    }

                    currentDocuments = documents;

                    listDiv.innerHTML = '<div class="docs-list">' + 
                        documents.map(doc => `
                            <div class="doc-item">
                                <div>
                                    <strong>${doc.filename}</strong><br>
                                    <small>Risk: <span class="risk-${doc.risk_level.toLowerCase()}">${doc.risk_level}</span> (${doc.risk_score}/100)</small>
                                </div>
                                <button onclick="getDocumentSummary('${doc.document_id}')">View Summary</button>
                            </div>
                        `).join('') + '</div>';

                } catch (error) {
                    console.error('Error fetching documents:', error);
                }
            }

            async function askQuestion() {
                const question = document.getElementById('questionInput').value.trim();

                if (!question) {
                    alert('Please enter a question');
                    return;
                }

                if (currentDocuments.length === 0) {
                    alert('Please upload a document first');
                    return;
                }

                const askBtn = document.getElementById('askBtn');
                const originalText = askBtn.textContent;
                askBtn.disabled = true;
                askBtn.innerHTML = 'Analyzing... <div class="loading"></div>';

                try {
                    const response = await fetch('/ask/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });

                    const result = await response.json();

                    const resultDiv = document.getElementById('answerResult');

                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <h3>💡 Answer:</h3>
                            <div style="white-space: pre-line; line-height: 1.6; margin: 15px 0;">
                                ${result.answer}
                            </div>
                            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;">
                                Confidence: ${(result.confidence * 100).toFixed(1)}% | 
                                Relevant sections: ${result.relevant_chunks} | 
                                Method: ${result.method}
                            </div>
                        `;
                        resultDiv.classList.add('show');
                    } else {
                        resultDiv.innerHTML = `<div class="status error">❌ Error: ${result.detail}</div>`;
                        resultDiv.classList.add('show');
                    }

                } catch (error) {
                    const resultDiv = document.getElementById('answerResult');
                    resultDiv.innerHTML = `<div class="status error">❌ Network error: ${error.message}</div>`;
                    resultDiv.classList.add('show');
                } finally {
                    askBtn.disabled = false;
                    askBtn.textContent = originalText;
                }
            }

            function askExample(question) {
                document.getElementById('questionInput').value = question;
                askQuestion();
            }

            async function getDocumentSummary(docId) {
                try {
                    const response = await fetch(`/summary/${docId}`);
                    const summary = await response.json();

                    if (response.ok) {
                        const resultDiv = document.getElementById('answerResult');
                        resultDiv.innerHTML = `
                            <h3>📋 Document Summary</h3>
                            <div style="margin: 15px 0;">
                                <strong>File:</strong> ${summary.filename}<br>
                                <strong>Type:</strong> ${summary.document_type}<br>
                                <strong>Risk Level:</strong> <span class="risk-${summary.risk_level.toLowerCase()}">${summary.risk_level}</span><br>
                                <strong>Risk Score:</strong> ${summary.risk_score}/100
                            </div>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0;">
                                ${summary.risk_summary}
                            </div>
                            ${summary.key_concerns.length > 0 ? `
                                <h4>🔍 Key Concerns:</h4>
                                <ul style="margin: 10px 0; padding-left: 20px;">
                                    ${summary.key_concerns.map(concern => `<li>${concern}</li>`).join('')}
                                </ul>
                            ` : ''}
                        `;
                        resultDiv.classList.add('show');
                    }
                } catch (error) {
                    console.error('Error getting summary:', error);
                }
            }

            function showStatus(message, type) {
                const statusDiv = document.getElementById('uploadStatus');
                statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            }

            // Load documents on page load
            window.onload = refreshDocuments;
        </script>
    </body>
    </html>
    """

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and analyze a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Analyze document
            result = analyzer.add_document(temp_file_path)

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            return {
                "status": "success",
                "document_id": result["document_id"],
                "filename": file.filename,
                "risk_analysis": result["risk_analysis"],
                "chunks_created": result["chunks_created"]
            }

        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about uploaded documents"""
    try:
        result = analyzer.ask_question(request.question, request.document_id)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return QuestionResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            method=result["method"],
            relevant_chunks=result["relevant_chunks"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """Get list of uploaded documents"""
    try:
        return analyzer.list_documents()
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/summary/{document_id}")
async def get_document_summary(document_id: str):
    """Get comprehensive summary of a document"""
    try:
        result = analyzer.get_document_summary(document_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "documents_loaded": len(analyzer.documents),
            "analyzer_ready": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Simple Legal Document Analyzer")
    print("=" * 50)
    print("📄 Upload PDFs and get easy legal analysis")
    print("🌐 Web interface: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("💾 Memory usage: <900MB RAM")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
