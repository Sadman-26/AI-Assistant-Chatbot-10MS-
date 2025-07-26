# ================================================================
# COMPLETE BANGLA EDUCATIONAL RAG SYSTEM
# World-Class AI Engineering Implementation
# ================================================================

print("🚀 Initializing Bangla Educational RAG System...")
print("=" * 60)

# ================================================================
# STEP 1: Import Required Libraries
# ================================================================
print("\n📚 Importing required libraries...")

import os
import sys
import warnings
import glob
import json
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
# Core libraries
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Text processing
import re
import unicodedata
from collections import defaultdict

# LangChain components
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

# Vector database
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

print("✅ All libraries imported successfully!")

# ================================================================
# STEP 2: Configuration and Constants
# ================================================================
print("\n⚙️ Setting up configuration...")

class Config:
    """Configuration class for the RAG system"""
    
    # Paths
    DOC_FOLDER = "doc"  # Your document folder
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Model settings
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Better model for Bangla
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    
    # Retrieval settings
    TOP_K_DOCS = 5
    
    # Pinecone settings
    INDEX_NAME = "rag"
    DIMENSION = 1024  #embedder model dimension
    METRIC = "cosine"
    
    # Language settings
    PRIMARY_LANGUAGE = "bangla"
    SECONDARY_LANGUAGE = "english"

config = Config()

# ================================================================
# STEP 3: API Keys Setup
# ================================================================
print("\n🔑 Setting up API keys...")

load_dotenv()

# Get API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Set environment variables

print("✅ API keys configured!")

# ================================================================
# STEP 4: Bangla Text Processing Utilities
# ================================================================
print("\n🔤 Setting up Bangla text processing utilities...")

class BanglaTextProcessor:
    """Advanced Bangla text processing utilities"""
    
    @staticmethod
    def clean_bangla_text(text: str) -> str:
        """Clean and normalize Bangla text"""
        if not text:
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove unwanted characters but keep Bangla punctuation
        text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\s\.\,\!\?\;\:\'\"\(\)\-]', '', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s*([।,!?;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
        """Extract metadata from filename patterns"""
        filename = Path(filename).stem
        
        # Common patterns for educational files (optional)
        patterns = {
            'subject': r'(বাংলা|ইংরেজি|গণিত|বিজ্ঞান|সমাজ|ইতিহাস|ভূগোল)',
            'class': r'(class|শ্রেণি)[\s\-_]*(\d+|এক|দুই|তিন|চার|পাঁচ|ছয়|সাত|আট|নয়|দশ)',
            'chapter': r'(chapter|অধ্যায়)[\s\-_]*(\d+)',
            'book': r'(book|বই)[\s\-_]*(.+)',
        }
        
        metadata = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                if key in ['class', 'chapter']:
                    metadata[key] = match.group(2)
                else:
                    metadata[key] = match.group(1)
        
        metadata['filename'] = filename
        return metadata
    
    @staticmethod
    def detect_content_type(text: str) -> str:
        """Detect the type of educational content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['কবিতা', 'poem', 'ছন্দ', 'রাইম']):
            return 'poetry'
        elif any(word in text_lower for word in ['গল্প', 'story', 'উপন্যাস', 'novel']):
            return 'literature'
        elif any(word in text_lower for word in ['ব্যাকরণ', 'grammar', 'বানান', 'spelling']):
            return 'grammar'
        elif any(word in text_lower for word in ['ইতিহাস', 'history', 'historical']):
            return 'history'
        elif any(word in text_lower for word in ['বিজ্ঞান', 'science', 'scientific']):
            return 'science'
        elif any(word in text_lower for word in ['গণিত', 'math', 'mathematics']):
            return 'mathematics'
        else:
            return 'general'

text_processor = BanglaTextProcessor()

# ================================================================
# STEP 5: Enhanced Multilingual Embeddings
# ================================================================
print("\n🧠 Setting up Enhanced Multilingual Embeddings...")

class EnhancedBanglaEmbeddings(Embeddings):
    """
    Enhanced embedding class optimized for Bangla educational content
    """
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Optimize for multilingual performance
        self.model.max_seq_length = 512
        
        print("✅ Enhanced Bangla embedding model loaded successfully!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with optimized prefixes for educational content"""
        # Clean texts first
        cleaned_texts = [text_processor.clean_bangla_text(text) for text in texts]
        
        # Add educational passage prefix for better retrieval
        prefixed_texts = [f"passage: {text}" for text in cleaned_texts]
        
        # Generate embeddings with normalization
        embeddings = self.model.encode(
            prefixed_texts, 
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with educational context"""
        # Clean query text
        cleaned_text = text_processor.clean_bangla_text(text)
        
        # Add educational query prefix
        prefixed_text = f"query: {cleaned_text}"
        
        # Generate embedding
        embedding = self.model.encode(
            prefixed_text, 
            normalize_embeddings=True
        )
        return embedding.tolist()

# Initialize embeddings
embeddings = EnhancedBanglaEmbeddings()

# ================================================================
# STEP 6: Document Loader and Processor
# ================================================================
print("\n📄 Setting up document loader and processor...")

class BanglaDocumentLoader:
    """Advanced document loader for Bangla educational content"""
    
    def __init__(self, doc_folder: str = config.DOC_FOLDER):
        self.doc_folder = doc_folder
        self.supported_formats = ['.txt', '.md']
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the doc folder with enhanced metadata"""
        documents = []
        
        if not os.path.exists(self.doc_folder):
            print(f"⚠️ Creating doc folder: {self.doc_folder}")
            os.makedirs(self.doc_folder)
            print("📝 Please add your Bangla educational txt files to the 'doc' folder")
            return documents
        
        # Get all text files
        file_patterns = [f"**/*{ext}" for ext in self.supported_formats]
        all_files = []
        
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(self.doc_folder, pattern), recursive=True))
        
        print(f"📚 Found {len(all_files)} files to process")
        
        for file_path in all_files:
            try:
                # Load document with proper encoding
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Clean content
                cleaned_content = text_processor.clean_bangla_text(content)
                
                # Extract metadata
                base_metadata = text_processor.extract_metadata_from_filename(file_path)
                
                # Enhanced metadata
                metadata = {
                    **base_metadata,
                    'source': file_path,
                    'file_size': len(content),
                    'content_type': text_processor.detect_content_type(content),
                    'language': 'bangla' if self._is_bangla_content(content) else 'mixed',
                    'processed_at': datetime.now().isoformat(),
                    'char_count': len(cleaned_content),
                    'word_count': len(cleaned_content.split())
                }
                
                # Create document
                doc = Document(
                    page_content=cleaned_content,
                    metadata=metadata
                )
                documents.append(doc)
                
                print(f"✅ Processed: {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {str(e)}")
                continue
        
        print(f"✅ Successfully loaded {len(documents)} documents")
        return documents
    
    def _is_bangla_content(self, text: str) -> bool:
        """Check if text is primarily in Bangla"""
        bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(re.findall(r'[^\s\d\W]', text))
        return bangla_chars > (total_chars * 0.6) if total_chars > 0 else False

# ================================================================
# STEP 7: Advanced Text Splitter
# ================================================================
print("\n✂️ Setting up advanced text splitter...")

class BanglaTextSplitter(RecursiveCharacterTextSplitter):
    """Custom text splitter optimized for Bangla educational content"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        # Bangla-specific separators
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
        
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

# ================================================================
# STEP 8: Load and Process Documents
# ================================================================
print("\n📖 Loading and processing documents...")

# Initialize document loader
doc_loader = BanglaDocumentLoader()

# Load documents
documents = doc_loader.load_documents()

if not documents:
    print("⚠️ No documents found. Creating sample documents for demonstration...")
    
    # Create sample educational content
    sample_docs = [
        Document(
            page_content="""
            বাংলা ভাষা ও সাহিত্য

            বাংলা ভাষা দক্ষিণ এশিয়ার একটি গুরুত্বপূর্ণ ভাষা। এটি বাংলাদেশের রাষ্ট্রভাষা এবং ভারতের পশ্চিমবঙ্গ রাজ্যের দাপ্তরিক ভাষা।

            বাংলা সাহিত্যের ইতিহাস অত্যন্ত সমৃদ্ধ। রবীন্দ্রনাথ ঠাকুর, নজরুল ইসলাম, বঙ্কিমচন্দ্র চট্টোপাধ্যায় প্রমুখ মহান সাহিত্যিকদের অবদানে বাংলা সাহিত্য বিশ্বমানের।

            ব্যাকরণ:
            বাংলা ব্যাকরণে বিশেষ্য, সর্বনাম, বিশেষণ, ক্রিয়া, অব্যয় - এই পাঁচটি মূল পদ রয়েছে।
            """,
            metadata={
                'subject': 'বাংলা',
                'content_type': 'literature',
                'language': 'bangla',
                'source': 'sample_bangla_literature.txt'
            }
        ),
        Document(
            page_content="""
            ছন্দ ও কবিতা

            ছন্দ হলো কবিতার প্রাণ। বাংলা ছন্দের তিনটি প্রধান ভাগ:
            ১. অক্ষরবৃত্ত ছন্দ
            ২. মাত্রাবৃত্ত ছন্দ  
            ৩. স্বরবৃত্ত ছন্দ

            রবীন্দ্রনাথের বিখ্যাত কবিতা:
            "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।
            চিরদিন তোমার আকাশ, তোমার বাতাস, আমার প্রাণে বাজায় বাঁশি।"

            এটি আমাদের জাতীয় সংগীত।
            """,
            metadata={
                'subject': 'বাংলা',
                'content_type': 'poetry',
                'language': 'bangla',
                'source': 'sample_poetry.txt'
            }
        )
    ]
    documents = sample_docs

# Initialize text splitter
text_splitter = BanglaTextSplitter()

# Split documents into chunks
print(f"📝 Splitting {len(documents)} documents into chunks...")
text_chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(text_chunks)} text chunks")

# Display sample chunk info
if text_chunks:
    sample_chunk = text_chunks[0]
    print(f"\n📋 Sample chunk preview:")
    print(f"Content length: {len(sample_chunk.page_content)} characters")
    print(f"Metadata: {sample_chunk.metadata}")
    print(f"Content preview: {sample_chunk.page_content[:200]}...")

# ================================================================
# STEP 9: Initialize GROQ LLM
# ================================================================
print("\n🤖 Setting up Enhanced GROQ LLM...")

# Initialize the GROQ LLM with better model for Bangla
llm = ChatGroq(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
    timeout=60,
    max_retries=3
)

print("✅ Enhanced GROQ LLM initialized successfully!")

# ================================================================
# STEP 10: Setup Pinecone Vector Database
# ================================================================
print("\n🗄️ Setting up Enhanced Pinecone Vector Database...")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Check if index exists, create if not
existing_indexes = pc.list_indexes().names()
if config.INDEX_NAME not in existing_indexes:
    print(f"Creating new Pinecone index: {config.INDEX_NAME}")
    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.DIMENSION,
        metric=config.METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("✅ New Pinecone index created!")
else:
    print(f"✅ Using existing Pinecone index: {config.INDEX_NAME}")

# Get index reference
index = pc.Index(config.INDEX_NAME)

# ================================================================
# STEP 11: Create Vector Store and Retriever
# ================================================================
print("\n🔍 Creating enhanced vector store and retriever...")

# Check current vectors in index
stats = index.describe_index_stats()
total_vectors = stats.get('total_vector_count', 0)
print(f"Current vectors in index: {total_vectors}")

# Create or connect to vectorstore
vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=config.INDEX_NAME,
    embedding=embeddings
) if total_vectors == 0 else PineconeVectorStore(
    index_name=config.INDEX_NAME,
    embedding=embeddings
)

# Setup enhanced retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": config.TOP_K_DOCS}
)

print("✅ Enhanced vector store and retriever created!")

# ================================================================
# STEP 12: Bangla Educational Prompt Template
# ================================================================
print("\n📝 Setting up Bangla educational prompt template...")

# Enhanced system prompt for Bangla education
system_prompt = """
আপনি একজন অভিজ্ঞ এবং দক্ষ বাংলা শিক্ষক। আপনার দায়িত্ব হলো শিক্ষার্থীদের বাংলা ভাষা, সাহিত্য এবং সংস্কৃতি সম্পর্কে সঠিক ও সহায়ক তথ্য প্রদান করা।

## আপনার বিশেষত্ব:
- বাংলা ভাষা ও ব্যাকরণ
- বাংলা সাহিত্য ও কবিতা
- বাংলা সংস্কৃতি ও ঐতিহ্য
- শিক্ষামূলক বিষয়াবলী

## উত্তর প্রদানের নিয়মাবলী:

### ১. ভাষা ও স্টাইল:
- সর্বদা বাংলায় উত্তর দিন
- সহজ, সরল ও বোধগম্য ভাষা ব্যবহার করুন
- শিক্ষার্থীদের উপযোগী করে ব্যাখ্যা করুন
- প্রয়োজনে উদাহরণ দিন

### ২. তথ্যের নির্ভরযোগ্যতা:
- শুধুমাত্র প্রদত্ত context থেকে তথ্য ব্যবহার করুন
- অনুমানভিত্তিক তথ্য প্রদান করবেন না
- সূত্র উল্লেখ করুন যখন প্রয়োজন

### ৩. শিক্ষামূলক পদ্ধতি:
- ধাপে ধাপে ব্যাখ্যা করুন
- মূল বিষয়টি প্রথমে বলুন
- তারপর বিস্তারিত ব্যাখ্যা দিন
- প্রাসঙ্গিক উদাহরণ সহ উপস্থাপন করুন

### ৪. উত্তরের কাঠামো:
- স্পষ্ট ও সংগঠিত উত্তর দিন
- প্রয়োজনে পয়েন্ট আকারে লিখুন
- গুরুত্বপূর্ণ বিষয় তুলে ধরুন

### ৫. বিশেষ ক্ষেত্রে:
- কবিতা বা সাহিত্যের ক্ষেত্রে ভাব ও অর্থ ব্যাখ্যা করুন
- ব্যাকরণের ক্ষেত্রে নিয়ম ও উদাহরণ দিন
- ইতিহাস বা সংস্কৃতির ক্ষেত্রে প্রেক্ষাপট উল্লেখ করুন

### ৬. অজানা তথ্যের ক্ষেত্রে:
যদি প্রদত্ত তথ্যে উত্তর না থাকে, তাহলে বলুন:
"দুঃখিত, প্রদত্ত তথ্যে এই বিষয়ে যথেষ্ট তথ্য নেই। আরও নির্ভরযোগ্য সূত্র বা বিশেষজ্ঞের সাহায্য নিন।"

প্রসঙ্গ (Context): {context}

আপনার উত্তর অবশ্যই:
✓ বাংলায় হতে হবে
✓ শিক্ষামূলক হতে হবে  
✓ সঠিক ও নির্ভরযোগ্য হতে হবে
✓ সহজবোধ্য হতে হবে
"""

# Create enhanced prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

print("✅ Bangla educational prompt template configured!")

# ================================================================
# STEP 13: Create Enhanced RAG Chain
# ================================================================
print("\n🔗 Creating Enhanced RAG Chain...")

# Create document chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✅ Enhanced RAG chain created successfully!")

# ================================================================
# STEP 14: Advanced Query System
# ================================================================
print("\n🎯 Setting up Advanced Query System...")

class BanglaRAG:
    """Advanced Bangla Educational RAG System"""
    
    def __init__(self, rag_chain, retriever):
        self.rag_chain = rag_chain
        self.retriever = retriever
        self.query_history = []
    
    def query(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """
        Process educational query with enhanced features
        """
        print(f"\n🔍 প্রশ্ন প্রক্রিয়াকরণ: {question}")
        
        # Clean and process question
        cleaned_question = text_processor.clean_bangla_text(question)
        
        # Get response from RAG chain
        try:
            response = self.rag_chain.invoke({"input": cleaned_question})
            
            # Extract information
            answer = response.get("answer", "")
            source_docs = response.get("context", [])
            
            # Prepare result
            result = {
                "question": question,
                "answer": answer,
                "source_documents": source_docs,
                "num_sources": len(source_docs),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to history
            self.query_history.append({
                "question": question,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": "দুঃখিত, প্রশ্ন প্রক্রিয়াকরণে সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
                "source_documents": [],
                "num_sources": 0,
                "error": str(e)
            }
    
    def get_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents for a query"""
        try:
            docs = self.retriever.invoke(query)
            return docs[:k]
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_queries": len(self.query_history),
            "total_documents": len(documents),
            "total_chunks": len(text_chunks),
            "index_stats": index.describe_index_stats()
        }

# Initialize the RAG system
bangla_rag = BanglaRAG(rag_chain, retriever)





#Checking Results

print("✅ Advanced Bangla RAG System ready!")

# ================================================================
# STEP 15: Interactive Testing Interface
# ================================================================
print("\n🧪 System Testing & Demo Interface")
print("=" * 60)

# Display system stats
stats = bangla_rag.get_stats()
print(f"\n📊 System Statistics:")
print(f"   📚 Total Documents: {stats['total_documents']}")
print(f"   📄 Total Chunks: {stats['total_chunks']}")
print(f"   🔍 Vector Index: {stats['index_stats']}")

# Sample test queries
sample_queries = [
    "বাংলা ভাষার গুরুত্ব কী?",
    "রবীন্দ্রনাথ ঠাকুর সম্পর্কে বলুন।",
    "বাংলা ছন্দের প্রকারভেদ কী কী?",
    "জাতীয় সংগীতের কবি কে?",
    "বাংলা ব্যাকরণে কয়টি পদ আছে?"
]

print(f"\n📋 Sample Test Queries:")
for i, query in enumerate(sample_queries, 1):
    print(f"   {i}. {query}")

# Interactive query function
def ask_question(question: str = None):
    """Interactive question asking function"""
    if not question:
        question = input("\n💭 আপনার প্রশ্ন লিখুন (বাংলায়): ").strip()
    
    if not question:
        print("❌ Please enter a valid question")
        return
    
    print("\n" + "="*50)
    print(f"প্রশ্ন: {question}")
    print("="*50)
    
    # Get answer
    result = bangla_rag.query(question)
    
    # Display answer
    print(f"\n📝 উত্তর:")
    print("-" * 30)
    print(result['answer'])
    print("-" * 30)
    
    # Show sources if available
    if result.get('source_documents'):
        print(f"\n📚 তথ্যসূত্র ({result['num_sources']}টি ডকুমেন্ট):")
        for i, Updated_doc in enumerate(result['source_documents'][:3], 1):
            metadata = Updated_doc.metadata
            print(f"   {i}. {metadata.get('source', 'Unknown source')}")
            print(f"      {Updated_doc.page_content[:100]}...")
            print("-" * 30)

# ================================================================
# STEP 16: Interactive Testing Interface
# ================================================================
print("\n🧪 System Testing & Demo Interface")