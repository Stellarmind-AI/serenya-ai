import os
import io
import torch
import faiss
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import groq  # For Llama3 API

load_dotenv()
file_id = os.getenv("FILE_ID")
api_key = os.getenv("API_KEY")
print(api_key);
# === LOAD DATASET FROM GOOGLE DRIVE ===
download_url = f"https://drive.google.com/uc?id={file_id}"

response = requests.get(download_url)
if response.status_code != 200:
    raise ValueError("❌ Failed to download the dataset from Google Drive!")

df = pd.read_csv(io.StringIO(response.text))

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
CORS(app, resources={r"/": {"origins": "*"}})

# === INITIALIZE GROQ CLIENT ===
client = groq.Client(api_key=api_key)

# Function to search FAISS for similar dialogues
def search_faiss(query, top_k=3):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [text_data[i] for i in indices[0]]

# === FLASK ENDPOINT ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    retrieved_texts = search_faiss(user_query, top_k=3)
    context = "\n".join(retrieved_texts)

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an empathetic assistant."},
            {"role": "user", "content": f"Context:\n{context}\nUser Query: {user_query}"}
        ]
    )
    return jsonify({"response": response.choices[0].message.content})

# === RUN FLASK APP ===
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
