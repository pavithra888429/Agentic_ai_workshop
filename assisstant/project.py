import streamlit as st
import google.generativeai as genai
import requests

# 🔐 Configure Gemini API
genai.configure(api_key="AIzaSyBG4L1BaXBqZx2LsXWXc_mLLwUJgxjEFMk")  # Replace with your Gemini key
model = genai.GenerativeModel("gemini-2.0-flash")

# 🌐 Optional: n8n Webhook URL (will send data to this)
N8N_WEBHOOK_URL = "https://vishnupriya31.app.n8n.cloud/webhook-test/a0f96d35-36d4-4d20-ac11-ba14763ad8d2"  # Replace this with actual URL

# 🧠 Process Flow Generator
def generate_process_flow(problem_statement):
    prompt = f"""
You are an expert in Design Thinking and Product Development.

Given this problem statement:
"{problem_statement}"

Generate a complete Design Thinking process flow including:
1. Refined Problem Statement
2. Suggested AI Prototype or Solution
3. How to Test with Users
4. How to Collect & Integrate Feedback
5. Pitch to Stakeholders (in simple terms)
6. Final Launch Plan or Checklist

Each step should be clear and actionable.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# 🚀 Streamlit UI
st.title("🤖 AI-Powered Design Thinking Generator")
st.write("Enter a problem statement to auto-generate a full Design Thinking workflow.")

problem_input = st.text_area("📝 Problem Statement", placeholder="e.g., Students find it hard to manage time during exams")

if st.button("Generate Workflow"):
    if not problem_input.strip():
        st.warning("Please enter a problem statement first.")
    else:
        result = generate_process_flow(problem_input)
        st.subheader("🧠 Generated Process Flow")
        st.text(result)

        # 🔁 Send result to n8n webhook (if configured)
        try:
            response = requests.post(N8N_WEBHOOK_URL, json={
                "problem": problem_input,
                "process_flow": result
            })
            if response.status_code == 200:
                st.success("✅ Sent to n8n webhook successfully!")
            else:
                st.warning(f"⚠️ n8n webhook error: {response.status_code}")
        except Exception as e:
            st.warning(f"⚠️ Failed to send to n8n: {e}")
