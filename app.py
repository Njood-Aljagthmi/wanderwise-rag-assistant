import os
import tempfile
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================
# Load API Key Securely
# =========================
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning("OPENAI_API_KEY is missing. Please add it to your .env file.")

client = OpenAI(api_key=api_key) if api_key else None


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="WanderWise RAG Assistant",
    page_icon="✈️",
    layout="wide"
)


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 45%, #fffaf4 100%);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        padding: 28px 34px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.20);
        margin-bottom: 30px;
        text-align: center;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 0;
    }

    .upload-title {
        font-size: 26px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .upload-subtitle {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        margin-bottom: 16px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 34px;
        font-weight: 900;
    }

    .answer-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border-left: 6px solid #0284c7;
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        margin-top: 22px;
        margin-bottom: 24px;
    }

    .answer-title {
        font-size: 26px;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 12px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.4rem;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%);
        color: white;
        border: none;
    }

    .stTextInput input {
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        padding: 14px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        border-right: 1px solid #dbeafe;
    }

    .sidebar-title {
        font-size: 26px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 10px;
    }

    .sidebar-info {
        background: white;
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 16px;
        color: #475569;
        font-size: 13px;
        line-height: 1.7;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    .status-card {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        border-radius: 18px;
        padding: 16px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 16px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f8fbff 0%, #f0f9ff 100%);
        border: 2px dashed #93c5fd;
        border-radius: 20px;
        padding: 32px 22px;
        transition: 0.2s ease-in-out;
        margin-bottom: 28px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2563eb;
        background: #eff6ff;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        text-align: center;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
    }

    [data-testid="stFileUploaderDropzone"] small {
        color: #64748b !important;
        font-size: 13px !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
    }

    .streamlit-expanderHeader {
        font-weight: 700;
        color: #0f172a;
    }

    .neutral-note {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        color: #475569;
        font-size: 14px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Hero Section
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">WanderWise RAG Assistant ✈️🗺️</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# Default RAG Settings
# =========================
chunk_size = 800
overlap = 80
top_k = 3


# =========================
# Load Embedding Model
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


embedder = load_embedding_model()


# =========================
# File Readers
# =========================
def read_txt_or_md(file):
    return file.read().decode("utf-8", errors="ignore")


def read_pdf(file):
    text = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_path = temp_file.name

    reader = PdfReader(temp_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    os.remove(temp_path)
    return text


def load_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        file_name = file.name.lower()

        if file_name.endswith(".pdf"):
            text = read_pdf(file)
        elif file_name.endswith(".txt") or file_name.endswith(".md"):
            text = read_txt_or_md(file)
        else:
            continue

        if text.strip():
            documents.append({
                "file_name": file.name,
                "text": text
            })

    return documents


# =========================
# Chunking
# =========================
def chunk_text(text, chunk_size=800, overlap=80):
    chunks = []

    if not text:
        return chunks

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(documents, chunk_size=800, overlap=80):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["file_name"],
                "chunk_index": index
            })

    return all_chunks


# =========================
# FAISS Index
# =========================
def build_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# =========================
# Retrieval
# =========================
def retrieve(question, index, chunks, top_k=3):
    question_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(question_embedding, top_k)

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        retrieved.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": float(score)
        })

    return retrieved


# =========================
# Answer Generation
# =========================
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(
        [
            f"Source: {chunk['source']} | Chunk: {chunk['chunk_index']}\n{chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are WanderWise, a helpful travel RAG assistant.

Use only the provided travel context to answer the user's question.
Give a clear, practical, and friendly answer.
If the answer is not found in the uploaded documents, say:
"I could not find this information in the uploaded travel documents."

Context:
{context}

User question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a travel assistant that answers only from the provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# Upload Section
# =========================
st.markdown(
    """
    <div class="upload-title">Upload Travel Documents</div>
    <div class="upload-subtitle">
        Upload your travel guides, itineraries, or city notes in PDF, TXT, or MD format.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload your travel documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# =========================
# Process Uploaded Files
# =========================
documents = []
chunks = []

if uploaded_files:
    documents = load_documents(uploaded_files)
    chunks = create_chunks(documents, chunk_size=chunk_size, overlap=overlap)


# =========================
# Sidebar Overview
# =========================
st.sidebar.markdown('<div class="sidebar-title">✈️ WanderWise</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-info">
        <b>Travel Knowledge Assistant</b><br><br>
        Upload your travel documents and ask questions based on your own files.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Uploaded Files</div>
        <div class="metric-value">{len(uploaded_files) if uploaded_files else 0}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Processed Documents</div>
        <div class="metric-value">{len(documents)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Generated Chunks</div>
        <div class="metric-value">{len(chunks)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">Sources per Answer</div>
        <div class="metric-value">{top_k}</div>
    </div>
    """,
    unsafe_allow_html=True
)

if uploaded_files and chunks:
    st.sidebar.markdown(
        """
        <div class="status-card">
            ✅ Documents ready for search
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div class="sidebar-info">
            <b>Supported Files</b><br>
            PDF · TXT · MD
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Main App
# =========================
if uploaded_files and chunks:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Uploaded Documents</div>
                <div class="metric-value">{len(documents)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Generated Chunks</div>
                <div class="metric-value">{len(chunks)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Retrieved Sources</div>
                <div class="metric-value">{top_k}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    index = build_faiss_index(chunks)

    st.markdown(
        """
        <div class="upload-title" style="margin-top:20px;">Ask a Travel Question</div>
        <div class="upload-subtitle">
            Search across your uploaded documents and get an answer based on the most relevant sources.
        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Travel question",
        placeholder="Example: What can I do in Istanbul for 3 days?",
        label_visibility="collapsed"
    )

    if st.button("Ask WanderWise ✨"):
        if not api_key:
            st.error("Missing OPENAI_API_KEY. Add it first before generating answers.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your travel documents..."):
                retrieved_chunks = retrieve(question, index, chunks, top_k=top_k)
                answer = generate_answer(question, retrieved_chunks)

            answer_html = answer.replace("\n", "<br>")

            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-title">WanderWise Answer ✨</div>
                    <div style="font-size:16px; line-height:1.9; color:#1f2937;">
                        {answer_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Sources Used 📌")

            for i, chunk in enumerate(retrieved_chunks, 1):
                with st.expander(
                    f"Source {i}: {chunk['source']} | Chunk {chunk['chunk_index']} | Similarity: {chunk['score']:.3f}"
                ):
                    st.write(chunk["text"])
else:
    st.markdown(
        """
        <div class="neutral-note">
            Start by uploading one or more travel documents. Once uploaded, WanderWise will create chunks, build a vector index, and let you ask questions from your files.
        </div>
        """,
        unsafe_allow_html=True
    )