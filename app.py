import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import os
import pdfplumber #change 1

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (safe even if already installed)
nltk.download('stopwords')
nltk.download('wordnet')

# Define NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Load assets
# ---------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "nb_model3.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer_nb3.pkl"), "rb"))
label_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoder_nb3.pkl"), "rb"))
train_data = pickle.load(open(os.path.join(BASE_DIR, "training_resumes_nb3.pkl"), "rb"))

# Precompute training vectors for similarity
train_vectors = vectorizer.transform(train_data["clean_resume"])


# ---------------------------
# Improved Text Cleaning
# ---------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9+#. ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)


# ---------------------------
# Extract text from DOCX
# ---------------------------

def extract_docx(file):

    doc = Document(file)
    text = []

    for para in doc.paragraphs:
        text.append(para.text)

    return " ".join(text)

#change 2
def extract_pdf(file): 

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text

# ---------------------------
# Similarity calculation
# ---------------------------

def similarity_score(resume_vector):

    similarity = cosine_similarity(resume_vector, train_vectors)

    max_sim = similarity.max()

    return max_sim


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("AI Resume Classifier for HR")

st.write("Upload a resume to automatically classify job role and evaluate candidate strength.")

uploaded_files = st.file_uploader(
    "Upload Resumes (.docx)",
    type=["docx"],
    accept_multiple_files=True
)


if uploaded_files:

    results = []

    for uploaded_file in uploaded_files:

        st.divider()
        st.header(f"Resume: {uploaded_file.name}")

        # -----------------------
        # Extract
        # -----------------------

        if uploaded_file.name.endswith(".docx"):
            raw_text = extract_docx(uploaded_file)

        elif uploaded_file.name.endswith(".pdf"):
            raw_text = extract_pdf(uploaded_file)

        # Clean
        cleaned = clean_text(raw_text)

        # Vectorize
        vector = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vector)[0]
        role = label_encoder.inverse_transform([prediction])[0]

        # Probabilities
        probs = model.predict_proba(vector)[0]
        roles = label_encoder.classes_

        # Confidence score
        confidence = probs.max()

        # Similarity
        sim_score = similarity_score(vector)

        st.success(f"Predicted Role: **{role}**")

        col1, col2 = st.columns(2)

        col1.metric("Model Confidence", f"{confidence:.2f}")
        col2.metric("Similarity to Training Resumes", f"{sim_score:.2f}")

        # -----------------------
        # Confidence Graph
        # -----------------------

        st.subheader("Prediction Confidence by Role")

        fig, ax = plt.subplots()

        bars = ax.barh(roles, probs)

        for bar, r in zip(bars, roles):
            if r == role:
                bar.set_alpha(1.0)
            else:
                bar.set_alpha(0.4)

        ax.set_xlabel("Probability")
        ax.set_title("Role Classification Confidence")

        st.pyplot(fig)

        # -----------------------
        # Skill keyword analysis
        # -----------------------

        keywords = [
            "python","sql","machine learning","deep learning","nlp",
            "pandas","numpy","scikit","tensorflow","pytorch",
            "excel","power bi","tableau",
            "react","javascript","html","css",
            "java","spring","hibernate",
            "aws","azure","gcp","docker","kubernetes",
            "peopletools","peoplesoft","workday","fscm","hcm"
        ]

        found = []

        for k in keywords:
            if k.lower() in cleaned.lower():
                found.append(k)

        st.subheader("Detected Skills")

        if found:

            st.success("Skills detected in resume:")

            cols = st.columns(4)

            for i, skill in enumerate(found):
                cols[i % 4].write("✔ " + skill)

        else:
            st.warning("No major keywords detected")

        # -----------------------
        # Resume metrics
        # -----------------------

        word_count = len(cleaned.split())

        st.subheader("Resume Metrics")

        col3, col4 = st.columns(2)

        col3.metric("Word Count", word_count)
        col4.metric("Skill Keywords Found", len(found))

        # -----------------------
        # Candidate Fit Score
        # -----------------------

        st.subheader("Candidate Fit Evaluation")

        skill_score = len(found) / len(keywords)

        fit_score = (
            (confidence * 0.4) +
            (sim_score * 0.4) +
            (skill_score * 0.2)
        ) * 100

        fit_score = min(fit_score, 100)

        if fit_score >= 80:
            recommendation = "Strong Candidate"
        elif fit_score >= 60:
            recommendation = "Potential Fit"
        elif fit_score >= 40:
            recommendation = "Needs Review"
        else:
            recommendation = "Low Match"

        col7, col8 = st.columns(2)

        col7.metric("Candidate Fit Score", f"{fit_score:.0f} / 100")
        col8.metric("Hiring Recommendation", recommendation)

        st.progress(int(fit_score))
        results.append({
            "Candidate": uploaded_file.name,
            "Predicted Role": role,
            "Fit Score": round(fit_score,2),
            "Model Confidence": round(confidence,2),
            "Similarity Score": round(sim_score,2),
            "Skills Found": len(found),
            "Word Count": word_count,
            "Recommendation": recommendation
        })

        # -----------------------
        # Candidate strength
        # -----------------------

        percentile = sim_score * 100

        st.subheader("Candidate Strength vs Training Data")

        st.progress(int(percentile))

        st.write(
            f"This candidate is estimated to be stronger than **{int(percentile)}%** of resumes in the training set."
        )

        if st.button(f"View Clean Resume Text - {uploaded_file.name}"):

            st.subheader("Cleaned Resume Text")

            st.write(cleaned)

        st.write("Top probabilities")

        for r, p in zip(roles, probs):
            st.write(r, round(p, 3))

        # -----------------------
        # Candidate Ranking Table
        # -----------------------

    if results:

        summary_df = pd.DataFrame(results)

        # -----------------------
        # Overall Ranking Summary
        # -----------------------

        summary_df = summary_df.sort_values(
            by="Fit Score",
            ascending=False
        ).reset_index(drop=True)

        summary_df["Rank"] = summary_df.index + 1

        summary_df = summary_df[
            [
                "Rank",
                "Candidate",
                "Predicted Role",
                "Fit Score",
                "Model Confidence",
                "Similarity Score",
                "Skills Found",
                "Recommendation"
            ]
        ]

        st.divider()
        st.header("Candidate Ranking Summary")

        st.dataframe(summary_df, use_container_width=True)


        # -----------------------
        # Rank Candidates by Role
        # -----------------------

        st.divider()
        st.header("Candidates Ranked by Job Role")

        grouped_roles = summary_df.groupby("Predicted Role")

        best_candidates = []

        for role, group in grouped_roles:

            st.subheader(f"{role} Candidates")

            ranked = group.sort_values(
                by="Fit Score",
                ascending=False
            ).reset_index(drop=True)

            ranked["Rank"] = ranked.index + 1

            ranked = ranked[
                [
                    "Rank",
                    "Candidate",
                    "Fit Score",
                    "Model Confidence",
                    "Similarity Score",
                    "Skills Found",
                    "Recommendation"
                ]
            ]

            st.dataframe(ranked, use_container_width=True)

            best = ranked.iloc[0]

            best_candidates.append({
                "Role": role,
                "Best Candidate": best["Candidate"],
                "Fit Score": best["Fit Score"],
                "Confidence": best["Model Confidence"],
                "Skills": best["Skills Found"]
            })


        # -----------------------
        # HR Hiring Recommendations
        # -----------------------

        st.divider()
        st.header("HR Hiring Recommendations")

        best_df = pd.DataFrame(best_candidates)

        st.dataframe(best_df, use_container_width=True)


        # -----------------------
        # HR Explanation
        # -----------------------

        st.subheader("Hiring Explanation")

        for _, row in best_df.iterrows():

            st.write(
                f"✔ **{row['Best Candidate']}** is the strongest match for **{row['Role']}** "
                f"with a Fit Score of **{row['Fit Score']}**, strong model confidence, "
                f"and **{row['Skills']} detected technical skills**."
            )


        # -----------------------
        # HR Visualization
        # -----------------------

        st.subheader("Top Candidate Score by Role")

        fig3, ax3 = plt.subplots()

        ax3.bar(best_df["Role"], best_df["Fit Score"])

        ax3.set_ylabel("Fit Score")
        ax3.set_title("Best Candidate Per Role")

        st.pyplot(fig3)
