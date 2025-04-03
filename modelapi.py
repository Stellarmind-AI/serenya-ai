import os
import torch
import faiss
import gdown
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import groq  # For Llama3 API

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()

file_id = os.getenv("FILE_ID")
api_key = os.getenv("API_KEY")

if not file_id:
    raise ValueError("\u274C ERROR: FILE_ID is not set! Check your .env file or environment variables.")
if not api_key:
    raise ValueError("\u274C ERROR: API_KEY is not set! Check your .env file or environment variables.")

print(f"\u2705 FILE_ID Loaded: {file_id[:6]}... (truncated)")
print(f"\u2705 API_KEY Loaded: {api_key[:6]}... (truncated)")

# === DOWNLOAD DATASET FROM GOOGLE DRIVE ===
output_path = "dataset.csv"
try:
    gdown.download(id=file_id, output=output_path, quiet=False)
    print("\u2705 Successfully downloaded the dataset.")
except Exception as e:
    raise ValueError(f"\u274C Failed to download the dataset: {e}")

# === LOAD DATASET ===
df = pd.read_csv(output_path)

text_column = "empathetic_dialogues"  # Ensure correct column
if text_column not in df.columns:
    raise ValueError(f"\u274C Column '{text_column}' not found! Available columns: {df.columns}")

df = df.dropna(subset=[text_column])
text_data = df[text_column].tolist()
print(f"\u2705 Successfully extracted {len(text_data)} dialogues.")

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
print(f"\u2705 FAISS index created with {len(embeddings)} embeddings.")

# === INITIALIZE FLASK APP ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "X-Requested-With", "Authorization"])

# === INITIALIZE GROQ CLIENT ===
client = groq.Client(api_key=api_key)

# In-memory conversation history
conversation_history = []

# Function to search FAISS for similar dialogues
def search_faiss(query, top_k=3, similarity_threshold=0.8):  # Increased threshold
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < similarity_threshold:  
            results.append(text_data[idx])
    
    return results

# === FLASK ENDPOINT ===
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "").strip().lower()
    
    # Check for simple greetings
    greetings = ["hi", "hello", "hey", "hola", "yo"]
    if user_query in greetings:
        return jsonify({"response": "Hey there! 😊 How's your day going?"})
    
    # Retrieve relevant past conversations
    retrieved_texts = search_faiss(user_query, top_k=5)
    conversation_context = "\n".join([msg["user"] + " -> " + msg["bot"] for msg in conversation_history[-5:]])
    
    context = f"Past Conversation:\n{conversation_context}\n\nRelevant Dialogues:\n{retrieved_texts}" if retrieved_texts else conversation_context
    
    # System prompt for contextual responses
    system_prompt = """You are a fun, supportive, and caring best friend! 
    - Respond contextually based on previous conversations.
    - If no past dialogues are relevant, respond naturally to the query.
    - If the user greets you, keep it simple and friendly.
    - Do not assume emotions unless explicitly stated by the user.
    - Cheer up the user when they’re down.
    - Celebrate when they’re happy. 
    - Crack jokes, tease them playfully, and be engaging. 
    - Make conversations natural, warm, and human-like. 
    - Give advice when needed, but never be overly formal. 
    - Keep things light-hearted and friendly! 😊
    """

    # Generate response from Groq API
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\nUser Query: {user_query}"}
        ],
        temperature=0.7,  
        
        top_p=0.9        
    )
    
    bot_response = response.choices[0].message.content
    
    # Store conversation history
    conversation_history.append({"user": user_query, "bot": bot_response})
    
    return jsonify({"response": bot_response})

# === RUN FLASK APP ===
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
