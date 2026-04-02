import os
import uuid
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from unstructured.partition.pdf import partition_pdf
import pdfplumber

# -----------------------
# ENV
# -----------------------
load_dotenv()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Advanced RAG", layout="wide")
st.title("Advanced RAG (Layout aware + hierarchical sections level chunking  + Hybrid Retrieval + Reranking)")

# -----------------------
# SESSION STATE
# -----------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# -----------------------
# Embedding Wrapper
# -----------------------
from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0]

# -----------------------
# Load + Process PDF
# -----------------------
@st.cache_resource
def process_pdf(file):

    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.read())

    documents = []

    # 🔹 Unstructured parsing
    elements = partition_pdf(temp_path)
    for el in elements:
        if el.text:
            documents.append(
                Document(
                    page_content=el.text,
                    metadata={
                        "pageNo": el.metadata.page_number,
                        "source": el.metadata.filename,
                        "type": el.category
                    }
                )
            )

    # 🔹 Table extraction
    with pdfplumber.open(temp_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(
                    [" | ".join([str(cell) for cell in row if cell]) for row in table]
                )
                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={"type": "table", "page": page_num}
                    )
                )

    # 🔹 Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=700,
        chunk_overlap=100
    )

    def simple_abstract(text):
        return text[:200]

    section_data = []

    for doc in documents:
        chunks = text_splitter.split_documents([doc])

        section_data.append({
            "section_id": str(uuid.uuid4()),
            "full_text": doc.page_content,
            "abstract": simple_abstract(doc.page_content),
            "chunks": [c.page_content for c in chunks]
        })

    final_docs = []

    for sec in section_data:
        # Abstract
        final_docs.append(
            Document(
                page_content=sec["abstract"],
                metadata={"section_id": sec["section_id"], "type": "abstract"}
            )
        )

        # Chunks
        for chunk in sec["chunks"]:
            final_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"section_id": sec["section_id"], "type": "chunk"}
                )
            )

    return final_docs


# -----------------------
# Build RAG
# -----------------------
@st.cache_resource
def build_rag(final_docs):

    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = SentenceTransformerEmbeddings(embedding_model)

    vectorstore = FAISS.from_documents(final_docs, embedding)

    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    sparse_retriever = BM25Retriever.from_documents(final_docs)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.7]
    )

    return hybrid_retriever


# -----------------------
# Reranking
# -----------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def rerank_docs(query, docs):

    tokenized_docs = [tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    scored_docs = list(zip(docs, scores))
    reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in reranked[:5]]


# -----------------------
# LLM + Chain
# -----------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPEN_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

chat_template = ChatPromptTemplate.from_template(
    "Answer the question based only on the context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def generate_answer(docs, query):
    context = format_docs(docs)
    messages = chat_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


# -----------------------
# UI FLOW
# -----------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Step 1: Process
if uploaded_file and not st.session_state.processed:

    if st.button("📄 Process Document"):

        with st.spinner("Processing..."):
            final_docs = process_pdf(uploaded_file)
            retriever = build_rag(final_docs)

            st.session_state.retriever = retriever
            st.session_state.processed = True

        st.success("✅ Document processed successfully!")

# Step 2: Ask
if st.session_state.processed:

    st.success("🟢 System Ready")

    query = st.text_input("Enter your question")

    if st.button("🚀 Get Answer", disabled=not query):

        with st.spinner("Thinking..."):

            retrieved_docs = st.session_state.retriever.invoke(query)

            final_docs = rerank_docs(query, retrieved_docs)

            answer = generate_answer(final_docs, query)

        st.subheader("Answer")
        st.write(answer)

        # with st.expander("Retrieved Context"):
        #     for i, doc in enumerate(final_docs):
        #         st.write(f"### Doc {i+1}")
        #         st.write(doc.page_content)

# Reset
if st.button("🔄 Reset"):
    st.session_state.processed = False
    st.session_state.retriever = None