import streamlit as st
import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import base64
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import nltk
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
import os
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings
import pytesseract, pdfplumber, tempfile, os
from pdf2image import convert_from_path
import elasticsearch
from elasticsearch import Elasticsearch


ELASTICSEARCH_HOST = "https://your_elasticsearch_url"
ELASTICSEARCH_PORT = 0****
ELASTICSEARCH_USERNAME = "your_username"  # Replace with actual username
ELASTICSEARCH_PASSWORD = "your_password"  # Replace with actual password


es_client = Elasticsearch(
    f"{ELASTICSEARCH_HOST}",
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    request_timeout=60  # ⏰ increase timeout

)

def ocr_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error with PDFplumber: {e}. Attempting OCR fallback.")
        images = convert_from_path(file_path)
        for img in images:
            text += ocr_from_image(img)
    return text

def load_and_chunk(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

def process_pdf(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    raw_text = extract_text_from_pdf(tmp_path)
    chunks = load_and_chunk(raw_text)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ElasticsearchStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name="your_index",
        es_connection=es_client
    )

    os.remove(tmp_path)
    print("uploaded")
    return "✅ Το αρχείο προστέθηκε επιτυχώς στη βάση γνώσης."


# ---------------------------------------------------------
# ✅ Page config and styling
# ---------------------------------------------------------
st.set_page_config(page_title=" Multilingual Agent", page_icon="🌊")


def get_base64_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode()

def set_background_video_local(video_path):
    encoded_video = get_base64_video(video_path)
    video_tag = f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        video#bgvid {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
            object-fit: cover;
            opacity: 0.65;
        }}
        section.main > div.block-container {{
            padding-left: 350px;
            padding-right: 40px;
            max-width: 800px;
            z-index: 10;
            position: relative;
        }}
        .chat-bubble {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 1rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            font-size: 1.05rem;
            line-height: 1.6;
            font-family: "Segoe UI", sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .user-bubble {{
            background-color: rgba(230, 247, 255, 0.95);
            padding: 1rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            font-size: 1.05rem;
            line-height: 1.6;
            font-family: "Segoe UI", sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        </style>
        <video autoplay muted loop id="bgvid">
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
    """
    st.markdown(video_tag, unsafe_allow_html=True)


set_background_video_local("your_video_background.mp4")  # replace with your local file path


# ---------------------------------------------------------
# 🔐 Watsonx Credentials
# ---------------------------------------------------------
api_key = "your_WatsonX.ai_api_key"
project_id = "your_WatsonX.ai_project_id"
url = "your_model_deployment_url"
IAM_URL = "https://iam.cloud.ibm.com/identity/token"

# ---------------------------------------------------------
# 🎛️ Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Ρυθμίσεις")
st.sidebar.markdown("### 📄 Ανέβασε αρχείο PDF")
uploaded_pdf = st.sidebar.file_uploader("Επέλεξε ένα αρχείο PDF", type=["pdf"])
if uploaded_pdf:
    st.sidebar.success("✅ Το αρχείο φορτώθηκε!")
    # 🔜 Trigger your function here
    process_pdf(uploaded_pdf) 
decoding_method = st.sidebar.selectbox(
    "Μέθοδος Αποκωδικοποίησης",
    ["greedy", "sample"],
    help="Επέλεξε τρόπο παραγωγής απάντησης: 'greedy' για πιο σίγουρες απαντήσεις, 'sample' για πιο δημιουργικές."
)
temperature = st.sidebar.slider("Δημιουργικότητα", 0.0, 1.5, 0.7, 0.1, help="Ελέγχει τη δημιουργικότητα του μοντέλου. Χαμηλές τιμές → πιο προβλέψιμες απαντήσεις, υψηλές τιμές → πιο δημιουργικές και απρόβλεπτες απαντήσεις."
)
top_k = st.sidebar.slider("Top-k", 0, 100, 50, 5, help="Περιορίζει τις πιθανές λέξεις που επιλέγει το μοντέλο στις top-k πιο πιθανές. Μικρότερη τιμή σημαίνει πιο προβλέψιμες απαντήσεις."
)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05, help="Ενεργοποιεί δειγματοληψία πυρήνα (nucleus sampling), επιλέγοντας τις πιο πιθανές λέξεις μέχρι να καλυφθεί η πιθανότητα p. Π.χ., 0.9 σημαίνει ότι λαμβάνεται υπόψη το 90% των πιο πιθανών λέξεων."
)
max_tokens = st.sidebar.slider("Μέγιστος Αριθμός Λέξεων", 100, 2000, 1400, 100, help="Καθορίζει το μέγιστο μήκος της απάντησης που θα παραχθεί. Περισσότερα tokens σημαίνουν πιο εκτενείς απαντήσεις."
)
typing_effect = st.sidebar.checkbox("Εφέ πληκτρολόγησης", value=True)
if st.sidebar.button("🧹 Καθαρισμός Συνομιλίας"):
    st.session_state.messages = []
language = st.sidebar.radio("Γλώσσα απάντησης", options=["Ελληνικά", "Αγγλικά"])
if st.sidebar.button("Εξαγωγή Ιστορικού (TXT)"):
    history = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.get("messages", [])])
    st.download_button("Κατέβασε ιστορικό", data=history, file_name="chat_history.txt")

# ---------------------------------------------------------
# 💬 Embeddings and Vector Search
# ---------------------------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def rag_pipeline(user_query: str) -> str:
    query_vector = embedding_model.encode(user_query).tolist()

    es_client = Elasticsearch(
        "your_elasticsearch_url",
        basic_auth=("your_username", "your_password"),
        verify_certs=False,
        request_timeout=60
    )

    response = es_client.search(
        index="your_index_name",
        knn={"field": "vector", "query_vector": query_vector, "k": 5, "num_candidates": 100}
    )
    context = "\n\n".join([hit["_source"].get("text", "") for hit in response["hits"]["hits"]])

    chat_history = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history += f"\nΕρώτηση: {msg['content']}"
        elif msg["role"] == "assistant":
            chat_history += f"\nΑπάντηση: {msg['content']}"

    prompt = f"""Χρησιμοποίησε μόνο τις παρακάτω πληροφορίες για να απαντήσεις στην ερώτηση του χρήστη. Η απάντηση πρέπει να είναι μόνο στα {language}.

{chat_history}

Πληροφορίες:
{context}

Ερώτηση: {user_query}

Instructions
    -------------------------
    - Use only the above data to formulate your answer.
    - Pay attention to numbers, statistics and names.
    - Do not include any information not present in the data.
    - If the provided data is insufficient, indicate that you cannot find the answer.
    - Respond fully in {language}.
    - Write only the new generated text
    - Use bullets in order to improve readability

Απάντηση:"""

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}"
    token_res = requests.post(IAM_URL, headers=headers, data=data)
    token_res.raise_for_status()
    bearer_token = token_res.json()["access_token"]

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    if decoding_method == "greedy":
        params = {
            "decoding_method": decoding_method,
            "repetition_penalty": 1.2,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 50
        }
    else:
        params = {
            "decoding_method": decoding_method,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": 1.2,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 50
        }

    body = {
        "input": prompt,
        "parameters": params
    }


    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["results"][0]["generated_text"]

# ---------------------------------------------------------
# 🖊️ Typing Effect
# ---------------------------------------------------------
def simulate_typing(response_text, delay=0.02):
    output = ""
    placeholder = st.empty()
    for char in response_text:
        output += char
        placeholder.markdown(f'<div class="chat-bubble">{output}</div>', unsafe_allow_html=True)
        time.sleep(delay)
    return output

# ---------------------------------------------------------
# 💬 Chat UI
# ---------------------------------------------------------
st.markdown("<h1 style='text-align:center; color:white;'> Marketing Greece Assistant</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Καλωσήρθες!  Επέλεξε μια από τις παρακάτω ερωτήσεις για να ξεκινήσεις:"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        bubble = "user-bubble" if message["role"] == "user" else "chat-bubble"
        st.markdown(f'<div class="{bubble}">{message["content"]}</div>', unsafe_allow_html=True)

if len(st.session_state.messages) == 1:
    queries = [
        "Πρωτοβουλίες για sustainability που αφορά πολιτισμό και άυλη πολιτιστική κληρονομιά σε προορισμούς",
        "Tάσεις που να αφορούν ειδικά κοινά ταξιδιωτών",
        "Ενδιαφέρουσες δράσεις επικοινωνίας προορισμών σε niche κοινά",
        "Φέρε μου cases που να δείχνουν συνεργασίες σε τοπικό επίπεδο για τον προορισμό"
    ]
    for q in queries:
        if st.button(q):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("assistant"):
                with st.spinner(""):
                    st.markdown("<p style='color:white; font-weight:bold;'>Αναζητώ την απάντηση... ⏳</p>", unsafe_allow_html=True)
                    try:
                        response = rag_pipeline(q)
                        final = simulate_typing(response) if typing_effect else response
                        st.session_state.messages.append({"role": "assistant", "content": final})
                    except Exception as e:
                        st.error(f"⚠️ Σφάλμα: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Σφάλμα: {e}"})

user_input = st.chat_input("Πληκτρολόγησε την ερώτησή σου...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
            st.markdown("""
                    <div style='color:white; font-size:18px; font-weight:bold;'>
                    Αναζητώ την απάντηση
                    <span class="dots"><span>.</span><span>.</span><span>.</span></span>
                    </div>
                    <style>
                    @keyframes blink {
                    0% { opacity: 0.2; }
                    20% { opacity: 1; }
                    100% { opacity: 0.2; }
                    }
                    .dots span {
                    animation-name: blink;
                    animation-duration: 1.4s;
                    animation-iteration-count: infinite;
                    animation-fill-mode: both;
                    }
                    .dots span:nth-child(2) {
                    animation-delay: 0.2s;
                    }
                    .dots span:nth-child(3) {
                    animation-delay: 0.4s;  
                    }
                    </style>
                    """, unsafe_allow_html=True)
            try:
                response = rag_pipeline(user_input)
                final = simulate_typing(response) if typing_effect else response
                st.session_state.messages.append({"role": "assistant", "content": final})
            except Exception as e:
                error_msg = f"⚠️ Σφάλμα: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})