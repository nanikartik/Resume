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

uploaded_file = st.file_uploader(
    "Upload Resume (.docx or .pdf)",
    type=["docx","pdf"]
)#change 3

    keywords = [
        "python","sql","machine learning","deep learning","nlp",
        "pandas","numpy","scikit","tensorflow","pytorch",
        "excel","power bi","tableau",
        "react","javascript","html","css",
        "java","spring","hibernate",
        "aws","azure","gcp","docker","kubernetes",
        "peopletools","peoplesoft","workday","fscm","hcm"
        ]#change 5

if uploaded_file:

    # Extract (change 4)
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

    # -----------------------
    # Most Similar Resume(change 9 starts)
    # -----------------------
    
    similarity = cosine_similarity(vector, train_vectors)
    
    best_index = similarity.argmax()
    best_score = similarity.max()
    
    best_match = train_data.iloc[best_index]
    
    st.subheader("Most Similar Resume in Training Data")
    
    col5, col6 = st.columns(2)
    
    col5.metric("Matched Role", best_match["job_role"])
    col6.metric("Similarity Score", f"{best_score:.2f}")
    
    # Show top keywords from matched resume
    match_text = best_match["clean_resume"]
    
    match_keywords = []
    
    for k in keywords:
        if k.lower() in match_text.lower():
            match_keywords.append(k)
    
    if match_keywords:
        st.write("Common Skills With Matched Resume:")
        st.write(match_keywords[:8])#change 9 ends

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

    # highlight predicted class
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

    # keywords = [
    #     "python","sql","machine learning","deep learning","nlp",
    #     "pandas","numpy","scikit","tensorflow","pytorch",
    #     "excel","power bi","tableau",
    #     "react","javascript","html","css",
    #     "java","spring","hibernate",
    #     "aws","azure","gcp","docker","kubernetes",
    #     "peopletools","peoplesoft","workday","fscm","hcm"
    #     ]#change 5

    found = []

    for k in keywords:
        if k.lower() in cleaned.lower():
            found.append(k)#change 6

    st.subheader("Detected Skills")

    if found:
        st.success("Skills detected in resume:")
    
        cols = st.columns(4)

        for i, skill in enumerate(found):
            cols[i % 4].write("✔ " + skill)

    else:
        st.warning("No major keywords detected")#change 7


    # -----------------------
    # Resume metrics
    # -----------------------

    word_count = len(cleaned.split())

    st.subheader("Resume Metrics")

    col3, col4 = st.columns(2)

    col3.metric("Word Count", word_count)
    col4.metric("Skill Keywords Found", len(found))


    # -----------------------
    # Candidate strength
    # -----------------------

    percentile = sim_score * 100

    st.subheader("Candidate Strength vs Training Data")

    st.progress(int(percentile))

    st.write(f"This candidate is estimated to be stronger than **{int(percentile)}%** of resumes in the training set.")

    if st.button("View Clean Resume Text"):
        st.subheader("Cleaned Resume Text")
        st.write(cleaned) #change 8

    st.write("Top probabilities")

    for r,p in zip(roles, probs):
        st.write(r, round(p,3))
