import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data and model
df = pd.read_csv("data/qa_dataset.csv")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X = vectorizer.transform(df["question"])

# Streamlit UI
st.set_page_config(page_title="AI Tutor", page_icon="🧠")
st.title("🤖 AI Tutor – Learn with HOTA")

user_input = st.text_input("📘 Ask me a question:")

if user_input:
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, X).flatten()
    idx = sim_scores.argmax()
    confidence = sim_scores[idx]

    if confidence > 0.3:
        st.success(f"📗 Answer: {df.iloc[idx]['answer']}")
        st.caption(f"Confidence: {confidence:.2f}")
    else:
        st.warning("❓ I’m not sure yet... Try rephrasing your question.")
