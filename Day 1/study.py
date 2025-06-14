


# Initialize the Gemini model

import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

GOOGLE_API_KEY = "AIzaSyA8XcwcrV4Q_oQXXd6gp3LpdiJnGHPL-Do"

# Initialize Gemini 1.5 Flash model via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Prompt template to summarize material
summary_prompt = PromptTemplate.from_template("""
Summarize the following study material into 3–5 bullet points:

{material}
""")

# Prompt template to create multiple-choice questions
question_prompt = PromptTemplate.from_template("""
Based on the following summary, create 3 multiple-choice questions with 4 options each.
Mark the correct answer clearly after each question.

Summary:
{summary}
""")

# --- Streamlit UI ---
st.set_page_config(page_title="📘 Study Assistant", layout="wide")
st.title("📘 AI Study Assistant using LangChain + Gemini 1.5 Flash")

uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract text from PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            full_text += content

    st.subheader("📄 Extracted Text")
    st.text_area("Study Material", full_text, height=250)

    if st.button("📌 Generate Summary and Quiz"):
        with st.spinner("Generating summary..."):
            summary = llm.invoke(summary_prompt.format(material=full_text))

        st.subheader("📌 Summary")
        st.markdown(summary.content)

        with st.spinner("Generating questions..."):
            questions = llm.invoke(question_prompt.format(summary=summary.content))

        st.subheader("📝 Quiz Questions")
        st.markdown(questions.content)