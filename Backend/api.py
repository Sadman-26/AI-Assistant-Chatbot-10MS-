from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from qa_interface import ask_bangla_question

# Initialize FastAPI app
app = FastAPI(
    title="Bangla RAG Q&A API",
    description="A FastAPI wrapper for the Bangla Educational RAG Q&A system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    include_context: bool = False

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    success: bool = True
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return HealthResponse(
        status="healthy",
        message="Bangla RAG Q&A API is running. Use /ask endpoint to ask questions."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running successfully"
    )

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question in Bangla and get an answer using the RAG system.
    
    Args:
        request: QuestionRequest object containing the question and optional parameters
        
    Returns:
        QuestionResponse object with the answer and optional context
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get answer from the RAG system
        result = ask_bangla_question(request.question)
        
        # Extract answer and context
        answer = result[0]
        context = result[1] if request.include_context else None
        
        return QuestionResponse(
            answer=answer,
            context=context,
            success=True,
            message="Question answered successfully"
        )
        
    except Exception as e:
        return QuestionResponse(
            answer="",
            context=None,
            success=False,
            message=f"Error processing question: {str(e)}"
        )

@app.get("/docs")
async def get_documentation():
    """Get API documentation"""
    return {
        "title": "Bangla RAG Q&A API",
        "description": "API for asking questions in Bangla using RAG system",
        "endpoints": {
            "GET /": "Root endpoint with API information",
            "GET /health": "Health check endpoint",
            "POST /ask": "Ask a question in Bangla",
            "GET /docs": "This documentation"
        },
        "usage": {
            "ask_question": {
                "method": "POST",
                "endpoint": "/ask",
                "body": {
                    "question": "Your question in Bangla",
                    "include_context": "boolean (optional, default: false)"
                }
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 