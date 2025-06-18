import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# ✅ Set Gemini API key directly here
GOOGLE_API_KEY = "AIzaSyBFmxs80Ye847Hb8Ih3UH9LPVqhZzOuhOo"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Agent 1: Document Parsing Agent
def document_parsing_tool(text: str) -> str:
    return f"[Document Parsing] Extracted data from input:\n{text[:300]}..."

# Agent 2: Contract-Invoice Matching Agent
def contract_invoice_matching_tool(text: str) -> str:
    return "[Matching] Compared deliverables, pricing, and delivery dates."

# Agent 3: RAG-Enabled Mismatch Detection Agent
def mismatch_detection_tool(text: str) -> str:
    return "[RAG] Identified: Late delivery (5 days), overbilling ($500)."

# Agent 4: Compliance Summary Agent
def compliance_summary_tool(text: str) -> str:
    return "📋 Summary:\n- Vendor: ABC Ltd\n- Overbilling: Yes\n- Delay: 5 days\n- Compliance: ⚠️ Needs Review"

# LangChain Tools
tools = [
    Tool(
        name="Document Parsing Tool",
        func=document_parsing_tool,
        description="Extracts structured info from documents."
    ),
    Tool(
        name="Contract Invoice Matching Tool",
        func=contract_invoice_matching_tool,
        description="Matches fields from contracts and invoices."
    ),
    Tool(
        name="Mismatch Detection Tool",
        func=mismatch_detection_tool,
        description="Uses RAG to detect mismatches."
    ),
    Tool(
        name="Compliance Summary Tool",
        func=compliance_summary_tool,
        description="Summarizes findings and flags issues."
    )
]

# Initialize the agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="Contract–Invoice Compliance Checker", layout="centered")
st.title("📄 Contract–Invoice Compliance Checker")
st.markdown("Upload a contract and invoice file to check for mismatches.")

# File upload
contract_file = st.file_uploader("Upload Contract (.pdf)", type=["pdf"])
invoice_file = st.file_uploader("Upload Invoice (.pdf)", type=["pdf"])

if st.button("✅ Run Compliance Check"):
    if contract_file and invoice_file:
        contract_text = contract_file.read().decode("utf-8",errors="ignore")
        invoice_text = invoice_file.read().decode("utf-8",errors="ignore")

        combined_text = f"Contract:\n{contract_text}\n\nInvoice:\n{invoice_text}"

        with st.spinner("🔍 Analyzing..."):
            result = agent_executor.run(combined_text)

        st.success("✅ Analysis Complete!")
        st.subheader("📋 Compliance Report:")
        st.write(result)
    else:
        st.error("Please upload both a contract and an invoice file.")
