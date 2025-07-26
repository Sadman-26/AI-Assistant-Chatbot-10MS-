# AI Assistant Chatbot

A modern AI chatbot application with a Bangla RAG (Retrieval-Augmented Generation) Q&A system. The application features a beautiful React frontend with a FastAPI backend.

## Features

- 🤖 **AI-Powered Chat**: Ask questions in Bangla or English
- 🎨 **Modern UI**: Beautiful, responsive interface with dark/light theme
- 📱 **Mobile Responsive**: Works perfectly on all devices
- 🔄 **Real-time Chat**: Instant responses with loading indicators
- 📚 **RAG System**: Powered by advanced retrieval-augmented generation
- 🌐 **API Integration**: RESTful API with comprehensive documentation

## Project Structure

```
AI-Assistant-Chatbot/
├── Backend/           # FastAPI backend with RAG system
│   ├── api.py        # Main API server
│   ├── qa_interface.py
│   ├── rag.py
│   └── requirements.txt
├── Frontend/         # React + TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── pages/
│   └── package.json
└── package.json      # Root package.json for easy startup
```

## Prerequisites

- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **pip** (Python package manager)

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd AI-Assistant-Chatbot
```

2. **Install root dependencies**:
```bash
npm install
```

3. **Install frontend dependencies**:
```bash
cd Frontend
npm install
cd ..
```

4. **Install backend dependencies**:
```bash
cd Backend
pip install -r requirements.txt
cd ..
```

## Environment Setup

1. **Create environment file** in the Backend directory:
```bash
cd Backend
cp .env.example .env  # if .env.example exists
```

2. **Add your API keys** to the `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Running the Application

### Option 1: Run both backend and frontend together
```bash
npm run dev
```

### Option 2: Run separately

**Start the backend** (in one terminal):
```bash
npm run dev:backend
```

**Start the frontend** (in another terminal):
```bash
npm run dev:frontend
```

## Accessing the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Ask a Question
```bash
POST http://localhost:8000/ask
Content-Type: application/json

{
  "question": "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলুন",
  "include_context": false
}
```

## Development

### Frontend Development
- Built with React + TypeScript
- Uses Vite for fast development
- Styled with Tailwind CSS
- UI components from shadcn/ui

### Backend Development
- FastAPI server
- RAG system with Pinecone vector store
- Groq LLM integration
- CORS enabled for frontend communication

## Troubleshooting

### Common Issues

1. **Backend won't start**:
   - Check if Python and pip are installed
   - Verify all dependencies are installed: `pip install -r Backend/requirements.txt`
   - Check if API keys are set in `.env` file

2. **Frontend won't start**:
   - Check if Node.js is installed
   - Install dependencies: `cd Frontend && npm install`

3. **API connection issues**:
   - Ensure backend is running on port 8000
   - Check CORS settings in backend
   - Verify API endpoint in frontend services

4. **Module not found errors**:
   - Run `npm install` in Frontend directory
   - Check if all dependencies are properly installed

### Port Conflicts

If you get port conflicts:
- Backend: Change port in `Backend/api.py` (line 123)
- Frontend: Change port in `Frontend/vite.config.ts`

## Building for Production

```bash
npm run build
```

This will create a production build in the `Frontend/dist` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 