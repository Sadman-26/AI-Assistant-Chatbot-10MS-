# qa_interface.py
# Simple interface for Bangla Educational RAG Q&A

import os
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --- Configuration ---
INDEX_NAME = "rag"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TOP_K_DOCS = 5


load_dotenv()
# --- API Keys ---
# Get API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Embedding Loader ---
class SimpleBanglaEmbeddings(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

embeddings = SimpleBanglaEmbeddings()

# --- Load Pinecone Index ---
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(INDEX_NAME)

# --- Vector Store and Retriever ---
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_DOCS}
)

# --- Prompt Template ---
system_prompt = """
আপনি একজন অভিজ্ঞ এবং দক্ষ বাংলা শিক্ষক। আপনার দায়িত্ব হলো শিক্ষার্থীদের বাংলা ভাষা, সাহিত্য এবং সংস্কৃতি সম্পর্কে সঠিক ও সহায়ক তথ্য প্রদান করা।
✓ Answer should be in Bangla and AVOID THE MCQ TO GENERATE THE ANSWER. IT WILL BE A BIG MISTAKE.
✓ বাংলায় উত্তর দিন
✓ শিক্ষামূলক ও নির্ভরযোগ্য তথ্য দিন
✓ শুধুমাত্র প্রদত্ত context ব্যবহার করুন
✓ Strictly follow this rule: DON'T PROVIDE ANY OTHER TEXT THAN THE ANSWER. I WANT ONLY THE ANSWER. 
✓ Answer should be in Bangla and AVOID THE MCQ TO GENERATE THE ANSWER. IT WILL BE A BIG MISTAKE.

প্রসঙ্গ (Context): {context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# --- LLM and RAG Chain ---
# llm = ChatGroq(
#     model='meta-llama/llama-4-scout-17b-16e-instruct',
#     temperature=0.1,
#     max_tokens=2048,
#     timeout=60,
#     max_retries=3
# )

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

# import getpass
# import os

# if not os.environ.get("GOOGLE_API_KEY"):
#   os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# from langchain.chat_models import init_chat_model

# llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Q&A Function ---
def ask_bangla_question(question):
    response = rag_chain.invoke({"input": question})
    answer = response.get("answer", "")
    # Extract retrieved documents (context)
    context_docs = response.get("context", [])
    # Format context for display
    context_text = "\n\nপ্রসঙ্গ (Retrieved Context):\n"
    for i, doc in enumerate(context_docs, 1):
        context_text += f"ডকুমেন্ট {i}:\n{doc.page_content}\n{'-'*50}\n"
    return [answer, context_text]

if __name__ == "__main__":
    print("Bangla RAG Q&A Interface Ready! Type your question in Bangla.")
    while True:
        q = input("\nপ্রশ্ন: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit", "q"]:
            break
        ans = ask_bangla_question(q)
        print("\nউত্তর:\n", ans[0]) 
        print("===============================================================") 
        print("\nContext:\n", ans[1]) 