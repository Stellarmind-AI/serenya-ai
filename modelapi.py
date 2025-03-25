import os
import io
import torch
import faiss
import gdown
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import groq  # For Llama3 API

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

# Debugging: Check if FILE_ID and API_KEY are loaded
file_id = os.getenv("FILE_ID")
api_key = os.getenv("API_KEY")

if not file_id:
    raise ValueError("❌ ERROR: FILE_ID is not set! Check your .env file or environment variables.")
if not api_key:
    raise ValueError("❌ ERROR: API_KEY is not set! Check your .env file or environment variables.")

print(f"✅ FILE_ID Loaded: {file_id[:6]}... (truncated for security)")
print(f"✅ API_KEY Loaded: {api_key[:6]}... (truncated for security)")

# === DOWNLOAD DATASET FROM GOOGLE DRIVE ===
output_path = "dataset.csv"
try:
    gdown.download(id=file_id, output=output_path, quiet=False)
    print("✅ Successfully downloaded the dataset.")
except Exception as e:
    raise ValueError(f"❌ Failed to download the dataset: {e}")

# === LOAD DATASET ===
df = pd.read_csv(output_path)

# Ensure correct column name
text_column = "empathetic_dialogues"  # Update this if needed
if text_column not in df.columns:
    raise ValueError(f"❌ Column '{text_column}' not found! Available columns: {df.columns}")

df = df.dropna(subset=[text_column])
text_data = df[text_column].tolist()
print(f"✅ Successfully extracted {len(text_data)} dialogues.")

# === LOAD TOKENIZER & MODEL ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# === GENERATE & STORE EMBEDDINGS IN FAISS ===
embeddings = [get_embedding(text) for text in text_data]
embeddings = torch.tensor(embeddings).numpy()

d = embeddings.shape[1]  # Dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings)
print(f"✅ FAISS index created with {len(embeddings)} embeddings.")

# === INITIALIZE FLASK APP ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Requested-With", "Authorization"])

# === INITIALIZE GROQ CLIENT ===
client = groq.Client(api_key=api_key)

# Function to search FAISS for similar dialogues
def search_faiss(query, top_k=3, similarity_threshold=0.5):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < similarity_threshold:  # Include only relevant results
            results.append(text_data[idx])
    return results

# === FLASK ENDPOINT ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400

    retrieved_texts = search_faiss(user_query, top_k=5)
    context = "\n".join(retrieved_texts) if retrieved_texts else "No relevant past dialogues found."

    # System prompt for 
    # a friendly chatbot
    system_prompt = """You are a fun, supportive, and caring best friend! 
    - Cheer up the user when they’re down.
    - Celebrate when they’re happy. 
    - Crack jokes, tease them playfully, and be engaging. 
    - Make conversations natural, warm, and human-like. 
    - Give advice when needed, but never be overly formal. 
    - Keep things light-hearted and friendly! 😊"""

    # Call Groq API
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\nUser Query: {user_query}"}
        ],
        temperature=0.9,  # Increase randomness
        top_p=0.95         # Increase response diversity
    )
    
    return jsonify({"response": response.choices[0].message.content})

# === RUN FLASK APP ===
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
