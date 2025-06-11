import pandas as pd
import mlflow
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Q&A data
df = pd.read_csv("data/qa_dataset.csv")

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

# Start MLflow experiment
mlflow.set_experiment("AI-Tutor-Vectorizer")
with mlflow.start_run():
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_metric("num_questions", len(df))

    # Save vectorizer
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    mlflow.log_artifact("tfidf_vectorizer.pkl")

print("✅ Training complete and vectorizer saved.")

