import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
import os

# 🔑 Set your keys directly
GOOGLE_API_KEY = "AIzaSyDEFb8x24E3l91Oa7taFwe8wlIDfvjZBfM"
SERPAPI_API_KEY = "67283903ee235f7fae4f8fb95371c2a238c5991fc697e6a15637a4125a8f64be"

# 🔗 Gemini LLM & Embeddings
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.4,
    google_api_key=GOOGLE_API_KEY
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 📚 Load Vector Store
@st.cache_resource
def load_vectorstore():
    docs = []
    for file in os.listdir("rag_docs"):
        path = os.path.join("rag_docs", file)
        loader = PyPDFLoader(path) if file.endswith(".pdf") else TextLoader(path)
        docs.extend(loader.load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings).as_retriever()

# 1️⃣ Competitor Discovery
def run_competitor_discovery(project_metadata):
    tool = Tool(
        name="SearchCompetitors",
        func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run,
        description="Search for competitor apps"
    )
    agent = initialize_agent([tool], llm, agent="zero-shot-react-description", verbose=True)
    return agent.run(f"Find competitor apps for: {project_metadata}")

# 2️⃣ Feature Benchmarking
def run_feature_benchmarking(competitor_text):
    prompt = f"""
Compare these apps:
{competitor_text}
Return a markdown table with: App | Features | Pros | Cons | Rating
Summarize why users like/dislike each app.
"""
    return llm.invoke(prompt).content

# 3️⃣ Extract User Patterns
def run_user_patterns(benchmark_text):
    prompt = f"""
From this data:
{benchmark_text}
List:
1. Top 3 features users love
2. Top 3 pain points users report
Avoid app names. Focus on patterns.
"""
    return llm.invoke(prompt).content

# 4️⃣ Clarifying Questions via RAG
def run_clarifying_questions_rag(user_patterns, retriever):
    context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(user_patterns)[:3]])
    prompt = f"""
User patterns:
{user_patterns}

Industry insights:
{context}

Generate 10 clarifying questions to improve and differentiate a student product idea.
"""
    return llm.invoke(prompt).content

# 5️⃣ Opportunity Reframing
def run_opportunity_reframe(questions):
    prompt = f"""
Based on these questions:
{questions}
Return:
- One-line refined problem
- Opportunity summary
- 3 action steps for research
"""
    return llm.invoke(prompt).content

# 🔁 Master Pipeline
def run_pipeline(project_idea):
    retriever = load_vectorstore()
    step1 = run_competitor_discovery(project_idea)
    step2 = run_feature_benchmarking(step1)
    step3 = run_user_patterns(step2)
    step4 = run_clarifying_questions_rag(step3, retriever)
    step5 = run_opportunity_reframe(step4)
    return {
        "1. Competitors": step1,
        "2. Benchmarking": step2,
        "3. User Patterns": step3,
        "4. Clarifying Questions": step4,
        "5. Opportunity Reframe": step5
    }

# 🖥️ Streamlit UI
st.set_page_config(page_title="Agentic AI System", layout="wide")
st.title("🤖 Agentic AI System for Student Project Refinement")

with st.expander("ℹ️ About this App"):
    st.markdown("""
This AI system helps students analyze competitors, identify user preferences, and refine project scope using Gemini-powered agents and RAG.
""")

project = st.text_area("🎯 Enter your project idea", placeholder="e.g., An app to support student mental wellness during exams")

if st.button("🚀 Run Agentic AI Pipeline"):
    with st.spinner("Running agents..."):
        result = run_pipeline(project)
    for step, output in result.items():
        st.subheader(step)
        st.code(output)