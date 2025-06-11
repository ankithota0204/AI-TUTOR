import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load vectorizer and data
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer, tfidf_matrix, data = pickle.load(f)

st.title("📚 AI Tutor with Chat History")

# Session state to track Q&A
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
query = st.text_input("Ask a question 👇")

if st.button("Get Answer"):
    if query:
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, tfidf_matrix)
        idx = similarity.argmax()
        answer = data['answer'][idx]

        # Save to chat history
        st.session_state.chat_history.append((query, answer))

# Show history
if st.session_state.chat_history:
    st.markdown("### 📝 Chat History")
    for q, a in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
                # Save to chat history
        st.session_state.chat_history.append((query, answer))

        st.success(f"Answer: {answer}")

        feedback = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"], key=f"feedback_{len(st.session_state.chat_history)}")
        # Store or log this feedback for future analysis
        st.session_state.chat_history[-1] += (feedback,)
uploaded_file = st.file_uploader("Upload additional Q&A CSV", type=["csv"])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    data = pd.concat([data, new_data], ignore_index=True)
    tfidf_matrix = vectorizer.fit_transform(data['question'])  # Update the matrix
    st.success("New topics added! Ask away.")
