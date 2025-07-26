# Bangla RAG Q&A API

A FastAPI wrapper for the Bangla Educational RAG (Retrieval-Augmented Generation) Q&A system.

## Features

- **RESTful API**: Easy-to-use HTTP endpoints for asking questions in Bangla
- **RAG Integration**: Uses the existing RAG system with Pinecone vector store and Groq LLM
- **CORS Support**: Cross-origin resource sharing enabled for web applications
- **Health Checks**: Built-in health monitoring endpoints
- **Documentation**: Auto-generated API documentation
- **Error Handling**: Comprehensive error handling and response formatting

## Installation

1. Make sure you have all the required dependencies installed:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Running the API

### Method 1: Direct Python execution
```bash
python api.py
```

### Method 2: Using uvicorn directly
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: API information and status
- **Response**: Basic API information

### 2. Health Check
- **URL**: `GET /health`
- **Description**: Health check endpoint
- **Response**: API health status

### 3. Ask Question
- **URL**: `POST /ask`
- **Description**: Ask a question in Bangla and get an answer
- **Request Body**:
```json
{
    "question": "Your question in Bangla",
    "include_context": false
}
```
- **Response**:
```json
{
    "answer": "The answer in Bangla",
    "context": "Retrieved context (if include_context=true)",
    "success": true,
    "message": "Success message"
}
```

### 4. Documentation
- **URL**: `GET /docs`
- **Description**: API documentation
- **Response**: Detailed API documentation

## Interactive API Documentation

FastAPI automatically generates interactive API documentation. Visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Usage Examples

### Using curl

1. **Health Check**:
```bash
curl http://localhost:8000/health
```

2. **Ask a Question**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন",
       "include_context": false
     }'
```

3. **Ask with Context**:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "রবীন্দ্রনাথ ঠাকুরের জীবনী সম্পর্কে জানতে চাই",
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

### Using JavaScript (fetch)

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
askQuestion('বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন', false)
    .then(result => {
        console.log('Answer:', result.answer);
        if (result.context) {
            console.log('Context:', result.context);
        }
    });
```

## Testing

Run the test script to verify the API functionality:

```bash
python test_api.py
```

This will test all endpoints with sample Bangla questions.

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for the LLM
- `PINECONE_API_KEY`: Your Pinecone API key for vector storage

### API Configuration

You can modify the following in `api.py`:

- **Host and Port**: Change in the `uvicorn.run()` call
- **CORS Settings**: Modify the `CORSMiddleware` configuration
- **Response Format**: Modify the Pydantic models

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid question format or empty questions
- **500 Internal Server Error**: RAG system errors
- **Graceful Degradation**: Returns error messages instead of crashing

## Production Deployment

For production deployment:

1. **Security**: Configure CORS properly for your domain
2. **Environment**: Use proper environment variable management
3. **Logging**: Add proper logging configuration
4. **Monitoring**: Add health checks and monitoring
5. **Rate Limiting**: Consider adding rate limiting for the `/ask` endpoint

Example production configuration:

```python
# In api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `qa_interface.py` is in the same directory
2. **API Key Error**: Verify your environment variables are set correctly
3. **Port Already in Use**: Change the port in the uvicorn configuration
4. **CORS Issues**: Check the CORS configuration for your frontend domain

### Debug Mode

Run with debug logging:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## License

This API wrapper is part of the Bangla RAG Q&A system. 