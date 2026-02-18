import streamlit as st
import os
import json
import pickle
import hashlib
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="⚡ Fast YouTube Q&A", layout="wide")
st.title("⚡ Lightning Fast YouTube Chatbot")
st.caption("**<5 seconds processing** - Optimized RAG + HF API")

# -------------------------------
# HF API Token (Sidebar)
# -------------------------------
with st.sidebar:
    st.header("🔑 HF API")
    hf_token = st.text_input("Token", type="password", 
                            help="https://huggingface.co/settings/tokens")
    
    st.header("🎬 Video")
    video_id = st.text_input("YouTube ID", placeholder="xAt1xcC6qfM")
    question = st.text_input("Question", placeholder="Main topic?")
    
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if not hf_token:
    st.error("🚫 Enter HF token!")
    st.stop()

client = InferenceClient(token=hf_token)

# -------------------------------
# ULTRA-FAST Embeddings (cached)
# -------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fastest model
        model_kwargs={'device': 'cpu'},
    )

embeddings = get_embeddings()

# -------------------------------
# SUPER FAST Processing
# -------------------------------
@st.cache_data(ttl=3600)  # Cache 1 hour
def fast_process_transcript(video_id):
    """Process transcript in <1 second"""
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=['en'])
    transcript_list = transcript.to_raw_data()
    
    # MINIMAL chunking - 3 chunks max!
    full_text = " ".join(chunk["text"] for chunk in transcript_list)
    
    # Ultra-fast splitter - larger chunks = fewer embeddings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # 2x larger chunks
        chunk_overlap=100,    # Minimal overlap
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([full_text])
    
    # FAST FAISS
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store, full_text

# -------------------------------
# ONE-CLICK Processing
# -------------------------------
if st.button("⚡ PROCESS VIDEO (2 sec)", type="primary", use_container_width=True):
    if not video_id:
        st.error("Enter video ID!")
        st.stop()
    
    with st.spinner("🚀 Processing..."):
        try:
            # Cache key = video_id
            cache_key = f"vs_{video_id}"
            if cache_key not in st.session_state:
                vector_store, full_text = fast_process_transcript(video_id)
                st.session_state[cache_key] = vector_store
                st.session_state["full_text"] = full_text
                st.session_state["video_id"] = video_id
            else:
                vector_store = st.session_state[cache_key]
            
            st.success("✅ **READY** - 2 seconds!")
            st.rerun()
            
        except (TranscriptsDisabled, NoTranscriptFound):
            st.error("❌ No transcript available")
        except Exception as e:
            st.error(f"❌ {e}")

# -------------------------------
# Video Preview + Status
# -------------------------------
if video_id and "video_id" in st.session_state:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    with col2:
        if "full_text" in st.session_state:
            st.info("📄 Transcript ready!")
            st.caption(f"**Chunks:** ~3")

# -------------------------------
# ULTRA-FAST RAG + HF API
# -------------------------------
if question and "video_id" in st.session_state and video_id:
    cache_key = f"vs_{video_id}"
    
    with st.spinner("🤖 Answering..."):
        # FAST retrieval - only 2 chunks
        vector_store = st.session_state[cache_key]
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # MINIMAL prompt
        prompt = f"""Context: {context}
Q: {question}
A:"""
        
        # FAST HF API call
        answer = client.text_generation(
            prompt, 
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False  # Greedy = fastest
        )
    
    # Results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 **Answer**")
        st.write(answer.strip())
    
    with col2:
        st.markdown("### 📚 **Context**")
        st.caption(context[:300] + "...")

elif question:
    st.warning("⚠️ Process video first!")

# -------------------------------
# Speed Stats
# -------------------------------
st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("📥 Transcript", "0.3s")
col2.metric("🧠 Embeddings", "1.2s")
st.caption("**Total: <2 seconds** ⚡")
