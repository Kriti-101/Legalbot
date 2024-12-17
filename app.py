from fastapi import FastAPI
from pydantic import BaseModel
from model import create_legal_document_tool

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Print variables to ensure they are loaded
print("LangSmith API Key:", os.getenv("LANGCHAIN_API_KEY"))
print("LangSmith Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LangSmith Endpoint:", os.getenv("LANGCHAIN_ENDPOINT"))

# Access variables
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
print(f"API Key: {langchain_api_key}")


# Initialize FastAPI app
app = FastAPI()

# Specify your document path
document_path = "sample_legal_document.pdf"

# Create the LegalDocumentQA tool
qa_tool = create_legal_document_tool(document_path)

# Define input/output models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# Define the POST endpoint
@app.post("/ask-question/", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = qa_tool.func(request.question)  # Run the QA tool with the input question
    return AnswerResponse(answer=answer)

from fastapi.responses import HTMLResponse
from fastapi import Request

@app.get("/", response_class=HTMLResponse)
def main_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LegalDocumentQA</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f9;
                color: #333;
                text-align: center;
            }

            header {
                background-color: #4CAF50;
                color: white;
                padding: 1em 0;
                margin-bottom: 20px;
            }

            h1 {
                margin: 0;
            }

            .content {
                max-width: 800px;
                margin: auto;
                padding: 20px;
                background: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }

            a {
                color: #4CAF50;
                text-decoration: none;
                font-weight: bold;
            }

            a:hover {
                text-decoration: underline;
            }

            .form-container {
                margin-top: 20px;
                text-align: left;
            }

            label {
                font-weight: bold;
            }

            input[type="text"] {
                width: calc(100% - 22px);
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }

            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }

            button:hover {
                background-color: #45a049;
            }

            .output {
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: left;
            }

            footer {
                margin-top: 20px;
                font-size: 0.9em;
                color: #777;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>LegalDocumentQA</h1>
        </header>
        <div class="content">
            <p>Welcome to the LegalDocumentQA API!</p>
            <p>This service allows you to query legal documents and receive intelligent responses based on your questions.</p>
            <div class="form-container">
                <form id="qa-form" onsubmit="return handleFormSubmit(event)">
                    <label for="question">Enter your question:</label><br>
                    <input type="text" id="question" name="question" placeholder="Type your question here..." required><br>
                    <button type="submit">Get Answer</button>
                </form>
            </div>
            <div id="output" class="output" style="display: none;">
                <strong>Answer:</strong>
                <p id="answer-text"></p>
            </div>
        </div>
        <footer>
            <p>&copy; 2024 LegalDocumentQA Team</p>
        </footer>

        <script>
            async function handleFormSubmit(event) {
                event.preventDefault();
                const questionInput = document.getElementById('question');
                const outputDiv = document.getElementById('output');
                const answerText = document.getElementById('answer-text');

                const question = questionInput.value;

                try {
                    const response = await fetch('/ask-question/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        answerText.textContent = data.answer;
                        outputDiv.style.display = 'block';
                    } else {
                        answerText.textContent = 'Error: Unable to fetch the answer.';
                        outputDiv.style.display = 'block';
                    }
                } catch (error) {
                    answerText.textContent = 'Error: Unable to fetch the answer.';
                    outputDiv.style.display = 'block';
                }
            }
        </script>
    </body>
    </html>
    """
