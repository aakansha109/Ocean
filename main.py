import os
import logging
import sqlite3
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# -----------------------------
# Load environment variables
# -----------------------------
from dotenv import load_dotenv
load_dotenv()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to your files
FAISS_INDEX_FILE = os.path.join(script_dir, "argo_hierarchical_index.faiss")
METADATA_FILE = os.path.join(script_dir, "argo_metadata.csv")
SQLITE_FILE = os.path.join(script_dir, "argo_meta.db")
GEMINI_API_KEY = os.getenv("AIzaSyDIQFHCUM3Ywv6_Oeyp4zm_X7XbrCO58VE")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    filename="app.log",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Load FAISS + Metadata
# -----------------------------
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
metadata_df = pd.read_csv(METADATA_FILE)
metadata = {idx: row.to_dict() for idx, row in metadata_df.iterrows()}

# -----------------------------
# Utility Functions
# -----------------------------
def create_query_vector(query_text):
    # TODO: replace with real embedding model
    return np.random.rand(1, faiss_index.d).astype("float32")

def search_faiss(query_vector, top_k=5):
    distances, indices = faiss_index.search(query_vector, top_k)
    results = []
    for idx in indices[0]:
        if idx in metadata:
            results.append(metadata[idx])
    return results

def query_db(sql_query, params=()):
    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()
    cursor.execute(sql_query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

def call_gemini(prompt_text, max_tokens=300, temperature=0.5):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(
            prompt_text,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        raise

def generate_final_answer(user_query):
    try:
        query_vector = create_query_vector(user_query)
        faiss_results = search_faiss(query_vector)

        retrieved_text = "\n".join(result.get("summary", "No summary") for result in faiss_results)

        prompt = (
            f"User's question: {user_query}\n"
            f"Context:\n{retrieved_text}\n\n"
            "Provide a **short answer** (max 5-6 lines), focus on **numbers, stats, and key values**, "
            "avoid long explanations or generalizations. Be precise and concise."
        )

        return call_gemini(prompt, max_tokens=200)
    except Exception as e:
        logging.error(f"Error in generating answer: {e}")
        raise

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI(title="Argo Retrieval Assistant API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    try:
        answer = generate_final_answer(req.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error while processing query '{req.question}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")







