# Bangla RAG Q&A API Documentation

## Overview

The Bangla RAG Q&A API is a FastAPI-based service that provides intelligent question-answering capabilities for Bangla educational content. It uses a Retrieval-Augmented Generation (RAG) system with Pinecone vector store and OpenAI's GPT-4 model to provide accurate, context-aware answers in Bangla.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-backend-url.vercel.app`

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## Data Models

### QuestionRequest
```typescript
interface QuestionRequest {
  question: string;        // The question in Bangla
  include_context?: boolean; // Whether to include retrieved context (default: false)
}
```

### QuestionResponse
```typescript
interface QuestionResponse {
  answer: string;          // The generated answer in Bangla
  context?: string;        // Retrieved context documents (if include_context=true)
  success: boolean;        // Whether the request was successful
  message?: string;        // Additional message or error details
}
```

### HealthResponse
```typescript
interface HealthResponse {
  status: string;          // "healthy" or error status
  message: string;         // Status message
}
```

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns basic API information and status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Bangla RAG Q&A API is running. Use /ask endpoint to ask questions."
}
```

**Example:**
```bash
curl http://localhost:8000/
```

### 2. Health Check

**GET** `/health`

Checks if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running successfully"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### 3. Ask Question

**POST** `/ask`

Ask a question in Bangla and get an intelligent answer using the RAG system.

**Request Body:**
```json
{
  "question": "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন",
  "include_context": false
}
```

**Response:**
```json
{
  "answer": "বাংলা সাহিত্যের ইতিহাস অত্যন্ত সমৃদ্ধ এবং প্রাচীন। এটি প্রায় ১০০০ বছর আগে থেকে শুরু হয়েছে...",
  "context": null,
  "success": true,
  "message": "Question answered successfully"
}
```

**With Context:**
```json
{
  "question": "রবীন্দ্রনাথ ঠাকুরের জীবনী সম্পর্কে জানতে চাই",
  "include_context": true
}
```

**Response with Context:**
```json
{
  "answer": "রবীন্দ্রনাথ ঠাকুর (১৮৬১-১৯৪১) ছিলেন একজন বিশিষ্ট কবি, সাহিত্যিক এবং দার্শনিক...",
  "context": "প্রসঙ্গ (Retrieved Context):\nডকুমেন্ট 1:\nরবীন্দ্রনাথ ঠাকুরের জীবনী...\n--------------------------------------------------\n",
  "success": true,
  "message": "Question answered successfully"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "বাংলা ব্যাকরণের মূল নিয়মগুলি কী কী?",
       "include_context": false
     }'
```

### 4. API Documentation

**GET** `/docs`

Returns detailed API documentation.

**Response:**
```json
{
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
```

## Error Handling

### HTTP Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request (e.g., empty question)
- **500 Internal Server Error**: Server error or RAG system failure

### Error Response Format
```json
{
  "answer": "",
  "context": null,
  "success": false,
  "message": "Error processing question: [error details]"
}
```

## Frontend Integration

### TypeScript/JavaScript Usage

```typescript
import { apiService } from '@/services/api';

// Health check
const health = await apiService.healthCheck();
console.log(health.status); // "healthy"

// Ask a question
const response = await apiService.askQuestion({
  question: "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন",
  include_context: false
});

if (response.success) {
  console.log(response.answer);
} else {
  console.error(response.message);
}
```

### React Hooks Usage

```typescript
import { useAskQuestion, useHealthCheck } from '@/hooks/use-api';

// In your component
const askQuestionMutation = useAskQuestion();
const healthCheck = useHealthCheck();

// Ask a question
const handleAskQuestion = async () => {
  const response = await askQuestionMutation.mutateAsync({
    question: "বাংলা ব্যাকরণের নিয়ম",
    include_context: true
  });
  
  if (response.success) {
    console.log(response.answer);
    console.log(response.context);
  }
};

// Check API status
if (healthCheck.isLoading) {
  console.log("Checking API status...");
} else if (healthCheck.isError) {
  console.log("API is offline");
} else {
  console.log("API is online");
}
```

## RAG System Details

### Architecture

1. **Embedding Model**: `intfloat/multilingual-e5-large`
2. **Vector Store**: Pinecone
3. **LLM**: OpenAI GPT-4.1-mini
4. **Retrieval**: Top 5 most similar documents
5. **Language**: Bangla (বাংলা)

### System Prompt

The system uses a specialized prompt for Bangla educational content:

```
আপনি একজন অভিজ্ঞ এবং দক্ষ বাংলা শিক্ষক। আপনার দায়িত্ব হলো শিক্ষার্থীদের বাংলা ভাষা, সাহিত্য এবং সংস্কৃতি সম্পর্কে সঠিক ও সহায়ক তথ্য প্রদান করা।
✓ Answer should be in Bangla and AVOID THE MCQ TO GENERATE THE ANSWER. IT WILL BE A BIG MISTAKE.
✓ বাংলায় উত্তর দিন
✓ শিক্ষামূলক ও নির্ভরযোগ্য তথ্য দিন
✓ শুধুমাত্র প্রদত্ত context ব্যবহার করুন
✓ Strictly follow this rule: DON'T PROVIDE ANY OTHER TEXT THAN THE ANSWER. I WANT ONLY THE ANSWER.
```

### Configuration

- **Index Name**: "rag"
- **Top K Documents**: 5
- **Temperature**: 0.5
- **Max Retries**: 2

## Environment Variables

### Backend Required Variables

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Frontend Required Variables

```env
VITE_API_BASE_URL=https://your-backend-url.vercel.app
```

## CORS Configuration

The API is configured to allow all origins for development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production use.

## Monitoring

### Health Check Endpoint
Use `/health` endpoint to monitor API status.

### Frontend Status Indicator
The frontend includes a real-time status indicator that:
- Shows "Online" when API is accessible
- Shows "Offline" when API is unreachable
- Shows "Connecting..." during health checks
- Refreshes every 30 seconds

## Interactive Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Ask a simple question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "বাংলা সাহিত্যের ইতিহাস"}'

# Ask with context
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "রবীন্দ্রনাথ ঠাকুরের জীবনী",
       "include_context": true
     }'
```

### Using Python requests

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "বাংলা ব্যাকরণের মূল নিয়মগুলি কী কী?",
        "include_context": True
    }
)

if response.status_code == 200:
    result = response.json()
    print("Answer:", result["answer"])
    if result.get("context"):
        print("Context:", result["context"])
else:
    print("Error:", response.text)
```

### Using JavaScript fetch

```javascript
// Ask a question
async function askQuestion(question, includeContext = false) {
    const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            include_context: includeContext
        })
    });
    
    const result = await response.json();
    return result;
}

// Example usage
askQuestion("বাংলা সাহিত্যের ইতিহাস", true)
    .then(result => {
        if (result.success) {
            console.log("Answer:", result.answer);
            console.log("Context:", result.context);
        } else {
            console.error("Error:", result.message);
        }
    });
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Backend has CORS configured, check if backend is deployed
2. **404 Errors**: Ensure endpoint URLs are correct
3. **500 Errors**: Check backend logs and API key configuration
4. **Empty Responses**: Verify question is not empty and in Bangla

### Debug Information

The API includes detailed logging:
- Request/response logging
- Error details
- Health check status
- RAG system performance

Check browser console (F12) for frontend debug information and backend logs for server-side issues. 