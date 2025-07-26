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
AI-Assistant-Chatbot-10MS-/
├── Backend/           # FastAPI backend with RAG system
│   ├── api.py        # Main API server
│   ├── qa_interface.py
│   ├── rag.py
│   ├── requirement.txt
│   ├── env.example
│   ├── doc/          # Source documents for RAG
│   └── Updated_doc/  # Updated documents for RAG
├── Frontend/         # React + TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── pages/
│   ├── public/
│   ├── package.json
│   └── package-lock.json
└── README.md         # Project documentation
```

## Prerequisites

- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **pip** (Python package manager)

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd AI-Assistant-Chatbot-10MS-
```

2. **Install frontend dependencies**:
```bash
cd Frontend
npm install
cd ..
```

3. **Install backend dependencies**:
```bash
cd Backend
pip install -r requirement.txt
cd ..
```

## Environment Setup

1. **Create environment file** in the Backend directory:
```bash
cd Backend
cp env.example .env  # if env.example exists
```

2. **Add your API keys** to the `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Running the Application

### Start the backend (in one terminal):
```bash
cd Backend
uvicorn api:app --reload --port 8000
```

### Start the frontend (in another terminal):
```bash
cd Frontend
npm run dev
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
   - Verify all dependencies are installed: `pip install -r Backend/requirement.txt`
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
- Backend: Change port in `Backend/api.py`
- Frontend: Change port in `Frontend/vite.config.ts`

## Building for Production

```bash
cd Frontend
npm run build
```

This will create a production build in the `Frontend/dist` directory.

## 1. Text Extraction Method and Formatting Challenges

### Method Used
- **Library**: `TextLoader` and `DirectoryLoader` from LangChain
- **File Format**: Plain text files (`.txt`) with UTF-8 encoding
- **Processing**: Custom `BanglaDocumentLoader` class with enhanced metadata extraction

### Implementation Details
```python
class BanglaDocumentLoader:
    def __init__(self, doc_folder: str = "doc"):
        self.doc_folder = doc_folder
        self.supported_formats = ['.txt', '.md']
    
    def load_documents(self) -> List[Document]:
        # Load with proper UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
```

### Formatting Challenges Faced

#### Unicode Normalization
- **Challenge**: Bangla Unicode characters need proper normalization
- **Solution**: Used `unicodedata.normalize('NFC', text)`
- **Impact**: Ensures consistent character representation

#### Text Cleaning
```python
def clean_bangla_text(text: str) -> str:
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove unwanted characters but keep Bangla punctuation
    text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\s\.\,\!\?\;\:\'\"\(\)\-]', '', text)
    
    # Clean up punctuation spacing
    text = re.sub(r'\s*([।,!?;:])\s*', r'\1 ', text)
    
    return text.strip()
```

#### Content Structure Challenges
- **Mixed Content**: Documents contain Bangla-English mixed content
- **Educational Format**: Questions, answers, and explanations in structured format
- **Metadata Extraction**: Filename-based metadata extraction for educational context

## 2. Chunking Strategy

### Strategy Used
- **Method**: `RecursiveCharacterTextSplitter` with custom Bangla-specific separators
- **Chunk Size**: 1000 characters with 200 character overlap
- **Separators Priority**:

```python
separators = [
    "\n\n",  # Paragraph breaks
    "।\n",   # Bangla sentence end
    "।",     # Bangla sentence end
    "\n",    # Line breaks
    "।।",    # Double danda
    "।।।",   # Triple danda
    ". ",    # English sentence end
    "! ",    # Exclamation
    "? ",    # Question
    "; ",    # Semicolon
    ", ",    # Comma
    " ",     # Space
    ""       # Character level
]
```

### Why This Works Well for Semantic Retrieval

#### Semantic Coherence
- **Bangla Sentence Boundaries**: `।` preserves complete thoughts
- **Educational Context**: Paragraph breaks maintain topic continuity
- **Language-Specific**: Respects Bangla linguistic patterns

#### Overlap Strategy
- **200-character overlap**: Ensures context continuity across chunks
- **Context Preservation**: Maintains semantic relationships between chunks
- **Retrieval Quality**: Improves relevance of retrieved chunks

#### Educational Content Optimization
- **Question-Answer Pairs**: Preserves complete Q&A units
- **Topic Continuity**: Maintains educational topic flow
- **Structured Content**: Handles formatted educational material

## 3. Embedding Model Choice

### Model Used
- **Model**: `intfloat/multilingual-e5-large`
- **Dimensions**: 1024
- **Architecture**: Transformer-based multilingual model

### Why This Model

#### Multilingual Excellence
- **Designed for Multilingual Tasks**: Specifically optimized for languages including Bangla
- **Cross-lingual Understanding**: Handles Bangla-English mixed content effectively
- **Semantic Understanding**: Captures deep semantic relationships in educational content

#### Technical Advantages
```python
class EnhancedBanglaEmbeddings(Embeddings):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Add educational passage prefix for better retrieval
        prefixed_texts = [f"passage: {text}" for text in cleaned_texts]
        
        # Generate embeddings with normalization
        embeddings = self.model.encode(
            prefixed_texts, 
            normalize_embeddings=True,
            batch_size=32
        )
        return embeddings.tolist()
```



### Sample Queries and Outputs


#### Educational Content Suitability
- **Academic Vocabulary**: Pre-trained on diverse educational content
- **Concept Understanding**: Good at capturing educational concepts
- **Semantic Relationships**: Maps similar educational topics together

### How It Captures Meaning

#### Contextual Understanding
- **Transformer Architecture**: Uses attention mechanisms to understand word relationships
- **Semantic Similarity**: Maps similar concepts to nearby vector spaces
- **Cross-lingual Capability**: Handles Bangla-English mixed content effectively

#### Educational Focus
- **Concept Mapping**: Captures educational concept relationships
- **Topic Clustering**: Groups related educational content
- **Semantic Search**: Enables meaningful educational content retrieval

## 4. Similarity Method and Storage Setup

### Similarity Method
- **Algorithm**: Cosine similarity with normalized embeddings
- **Storage**: Pinecone vector database with serverless architecture

### Storage Configuration
```python
# Pinecone settings
INDEX_NAME = "rag"
DIMENSION = 1024  # embedder model dimension
METRIC = "cosine"

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric=METRIC,
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

### Why This Setup

#### Cosine Similarity Advantages
- **Normalized Vectors**: Best for semantic similarity with normalized embeddings
- **Semantic Focus**: Captures semantic rather than lexical similarity
- **Educational Content**: Effective for educational concept matching

#### Pinecone Benefits
- **Scalable**: Handles large document collections
- **Managed Service**: Reduces infrastructure complexity
- **Fast Retrieval**: Optimized for similarity search
- **Serverless**: Cost-effective for variable workloads

## 5. Query-Document Comparison Strategy

### Comparison Method

#### Prefix Strategy
```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    # Add educational passage prefix for better retrieval
    prefixed_texts = [f"passage: {text}" for text in cleaned_texts]
    return self.model.encode(prefixed_texts, normalize_embeddings=True)

def embed_query(self, text: str) -> List[float]:
    # Add educational query prefix
    prefixed_text = f"query: {cleaned_text}"
    return self.model.encode(prefixed_text, normalize_embeddings=True)
```

#### Retrieval Process
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_DOCS}  # Top 5 documents
)
```

### Handling Vague/Missing Context

#### Fallback Mechanisms
```python
def ask_bangla_question(question):
    try:
        response = rag_chain.invoke({"input": question})
        answer = response.get("answer", "")
        context_docs = response.get("context", [])
        
        if not answer or not context_docs:
            return ["দুঃখিত, প্রদত্ত তথ্যে এই বিষয়ে যথেষ্ট তথ্য নেই।", ""]
            
    except Exception as e:
        return ["দুঃখিত, প্রশ্ন প্রক্রিয়াকরণে সমস্যা হয়েছে।", ""]
```

#### Context Validation
- **Relevance Check**: Validates if retrieved documents are relevant
- **Empty Response Handling**: Provides appropriate fallback messages
- **Error Recovery**: Graceful degradation when retrieval fails

## 6. Results Relevance and Improvements

### Current Results Quality

#### Strengths
- **Semantic Understanding**: Good understanding of Bangla educational content
- **Appropriate Responses**: Relevant answers for well-defined questions
- **Language Handling**: Proper handling of Bangla grammar and literature
- **Educational Focus**: Tailored for educational content

#### Limitations
- **Document Quality**: Limited by available document quality and coverage
- **Generic Responses**: Some responses may be generic for complex questions
- **Chunk Dependencies**: Heavily depends on document chunk quality
- **Context Limitations**: May miss broader context in complex queries

### Potential Improvements

#### 1. Better Chunking
```python
# Semantic chunking based on topic boundaries
class SemanticBanglaTextSplitter:
    def split_by_topic(self, text: str) -> List[str]:
        # Split by educational topics
        topics = re.split(r'(?:^|\n)([A-Z][^।]*?।)', text)
        return topics

# Sentence-level splitting with transformers
from sentence_transformers import SentenceTransformer
def semantic_sentence_split(text: str) -> List[str]:
    # Use sentence transformers for better sentence detection
    pass
```

#### 2. Enhanced Embedding Model
```python
# Fine-tuning on Bangla educational content
def fine_tune_embeddings():
    # Fine-tune on domain-specific Bangla educational data
    pass

# Hybrid retrieval (dense + sparse)
def hybrid_retrieval(query: str):
    # Combine dense embeddings with sparse retrieval
    dense_results = dense_retriever.retrieve(query)
    sparse_results = sparse_retriever.retrieve(query)
    return combine_results(dense_results, sparse_results)
```

#### 3. Larger Document Collection
- **Comprehensive Literature**: Add more Bangla literature and educational content
- **Diverse Topics**: Include various educational subjects and topics
- **Structured Knowledge**: Add structured knowledge bases and ontologies
- **Quality Control**: Implement document quality assessment

#### 4. Advanced Retrieval
```python
# Re-ranking with cross-encoders
def rerank_with_cross_encoder(query: str, documents: List[Document]):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, doc.page_content) for doc in documents])
    return sort_by_scores(documents, scores)

# Query expansion for Bangla
def expand_bangla_query(query: str) -> List[str]:
    # Expand query with synonyms and related terms
    expanded_queries = []
    # Add Bangla synonyms
    # Add related educational terms
    return expanded_queries
```

#### 5. Context Enhancement
```python
# Document metadata filtering
def filter_by_metadata(query: str, metadata_filters: Dict):
    # Filter documents by subject, difficulty, content type
    pass

# Context window optimization
def optimize_context_window(query: str, documents: List[Document]):
    # Dynamically adjust context window based on query complexity
    pass

# Source credibility scoring
def score_source_credibility(document: Document) -> float:
    # Score documents based on source credibility
    # Consider factors like:
    # - Source authority
    # - Content quality
    # - Educational relevance
    # - Update frequency
    pass
```

### Implementation Roadmap

#### Phase 1: Immediate Improvements
1. **Enhanced Chunking**: Implement semantic chunking
2. **Query Expansion**: Add Bangla query expansion
3. **Error Handling**: Improve fallback mechanisms

#### Phase 2: Advanced Features
1. **Fine-tuned Embeddings**: Domain-specific model training
2. **Hybrid Retrieval**: Combine dense and sparse methods
3. **Re-ranking**: Implement cross-encoder re-ranking

#### Phase 3: Scalability
1. **Document Quality**: Implement quality assessment
2. **Source Credibility**: Add credibility scoring
3. **Performance Optimization**: Optimize for large-scale deployment

## Conclusion

The current RAG implementation demonstrates a solid understanding of RAG principles with thoughtful consideration for Bangla language specifics and educational content requirements. The system effectively handles:

- **Language-Specific Processing**: Proper Bangla text handling and chunking
- **Educational Content**: Tailored for educational Q&A scenarios
- **Semantic Understanding**: Good retrieval of relevant educational content
- **Error Handling**: Graceful degradation and appropriate fallbacks

The proposed improvements would significantly enhance the system's performance, making it more robust, accurate, and scalable for educational applications. 