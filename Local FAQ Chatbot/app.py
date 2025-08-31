import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
questions = []
answers = []
with open("data/faqs.txt", "r") as f:
    lines = f.read().split("\n")
    for i in range(0, len(lines), 2):
        q = lines[i].replace("Q: ", "")
        a = lines[i+1].replace("A: ", "")
        questions.append(q)
        answers.append(a)

# Create embeddings
question_embeddings = model.encode(questions, convert_to_numpy=True)

# Build FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Streamlit UI
st.title("🤖 Local MCP Server")
user_input = st.text_input("Ask your question:")

if user_input:
    query_vector = model.encode([user_input])
    D, I = index.search(query_vector, k=1)
    answer = answers[I[0][0]]
    st.write("**Answer:**", answer)
    
