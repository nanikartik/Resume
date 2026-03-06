import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import docx
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_model():

    model = joblib.load("nb_model.pkl")
    vectorizer = joblib.load("vectorizer_nb.pkl")
    label_encoder = joblib.load("label_encoder_nb.pkl")

    return model, vectorizer, label_encoder


def clean_text(text):

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)


def extract_text(file):

    try:

        # PDF
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)
                return text

        # DOCX
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return " ".join([p.text for p in doc.paragraphs])

        # DOC (basic fallback)
        elif file.name.endswith(".doc"):
            try:
                return file.read().decode(errors="ignore")
            except:
                return ""

    except Exception as e:
        st.error(f"File reading error: {e}")

    return ""


st.set_page_config(page_title="Resume Classifier", layout="wide")

st.title("📄 Resume Job Role Classifier")

st.markdown(
"""
Upload **PDF / DOC / DOCX** resume or paste resume text to predict the job role.
"""
)

model, vectorizer, label_encoder = load_model()

tab1, tab2 = st.tabs(["📁 Upload Resume", "✏️ Paste Text"])

resume_text = ""

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "doc", "docx"]
    )

    if uploaded_file:

        with st.spinner("Extracting text from resume..."):
            resume_text = extract_text(uploaded_file)

        if resume_text:
            st.success("Text extracted successfully")
            st.text_area("Extracted Resume Text", resume_text, height=200)

with tab2:

    resume_text = st.text_area(
        "Paste resume text here",
        height=250
    )


col1, col2 = st.columns([1,3])

with col1:

    if st.button("🚀 CLASSIFY", type="primary"):

        if resume_text.strip():

            with st.spinner("Analyzing resume..."):

                cleaned = clean_text(resume_text)

                vec = vectorizer.transform([cleaned])

                pred = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]

                role = label_encoder.inverse_transform([pred])[0]

                st.balloons()

                st.success(
                    f"🎯 Predicted Role: **{role.replace('_',' ').title()}**"
                )

                confidence = np.max(probs) * 100

                st.info(f"Confidence: {confidence:.2f}%")

                prob_df = pd.DataFrame({
                    "Role": label_encoder.classes_,
                    "Probability": probs
                }).set_index("Role")

                st.bar_chart(prob_df)

        else:
            st.error("Please upload a resume or paste text first.")


st.markdown("---")
st.caption("Resume classification system built with Streamlit.")
