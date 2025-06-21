import streamlit as st
import google.generativeai as genai
from langchain.agents import Tool, initialize_agent
from langchain.utilities import SerpAPIWrapper
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate

# 🔑 Directly enter your API keys here (NOT recommended for production)

SERPAPI_API_KEY = "67283903ee235f7fae4f8fb95371c2a238c5991fc697e6a15637a4125a8f64be"

# 🌐 Set up API keys
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    google_api_key="AIzaSyDaR6vA7_oChyxJ2p6JtVMbcn7AM5n9gkA"
)


# 🔍 SerpAPI tool
serp_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# 📄 Prompt templates
feature_prompt = PromptTemplate.from_template("""
You are a product analyst. Given these apps and their descriptions, create:
- A markdown table comparing features
- Bullet list of user likes and dislikes

Apps:
{apps}
""")

questions_prompt = PromptTemplate.from_template("""
Based on the following feature insights and user patterns, generate 10 clarifying questions to help a student team improve their innovation project.

Insights:
{insights}
""")

opportunity_prompt = PromptTemplate.from_template("""
You are an innovation coach. Reframe the following problem statement using the insights provided.

Original Problem:
{problem}

Insights:
{insights}

Output:
- 1-line refined problem statement
- 3 actionable opportunity directions
""")

# 🧠 Tool functions
def search_competitors(query: str) -> str:
    return serp_tool.run(f"{query} app competitors site:play.google.com")

def extract_features(apps: str) -> str:
    return llm.invoke(feature_prompt.format(apps=apps))

def generate_questions(insights: str) -> str:
    return llm.invoke(questions_prompt.format(insights=insights))

def reframe_problem(problem: str, insights: str) -> str:
    return llm.invoke(opportunity_prompt.format(problem=problem, insights=insights))

# 🔁 Agent runner
def run_agent(tool_func, description, input_text):
    tool = Tool(name="AgentTool", func=tool_func, description=description)
    agent = initialize_agent([tool], llm, agent="zero-shot-react-description", verbose=False)
    return agent.run(input_text)

# 🎛️ Streamlit UI
st.set_page_config(page_title="Gemini AI Agents", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent Competitor Analysis with Gemini 1.5 Flash")

with st.form("project_form"):
    title = st.text_input("Project Title", "Student Wellness App")
    theme = st.text_area("Theme", "Promoting mental health in college students")
    problem = st.text_area("Problem Statement", "Students struggle with stress, anxiety, and limited resources")
    users = st.text_input("Target User Group", "College students aged 18–25")
    submitted = st.form_submit_button("🔍 Run Analysis")

if submitted:
    st.session_state["problem"] = problem
    metadata = f"{title}, {theme}, {users}, {problem}"

    st.markdown("### 🔍 Step 1: Competitor Discovery")
    competitors = run_agent(search_competitors, "Searches apps", f"{title} {theme}")
    st.markdown(competitors)

    st.markdown("### 🧩 Step 2: Feature Benchmarking")
    features = run_agent(
        extract_features,
        "Analyzes features and user sentiment from app descriptions",
        competitors
    )
    st.markdown(features)

    st.markdown("### ❓ Step 3: Clarifying Questions")
    questions = run_agent(
        generate_questions,
        "Generates clarifying questions from feature insights",
        features
    )
    st.markdown(questions)

    st.markdown("### 🚀 Step 4: Opportunity Reframing")
    reframed = run_agent(
        lambda insights: reframe_problem(problem, insights),
        "Refines the problem statement using insights",
        features
    )
    st.markdown(reframed)

    st.success("✅ Analysis complete using Gemini 1.5 Flash!")
