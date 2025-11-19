import streamlit as st
import requests
import json

# Gemini API Config (securely read from Streamlit secrets)
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
API_KEY = st.secrets["GEMINI_API_KEY"]

def gemini_call(system_prompt, user_query, context=None):
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": user_query})
    data = {"contents": [{"parts": [{"text": json.dumps(messages)}]}]}
    response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=data)
    if response.status_code == 200:
        outputs = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return outputs[0]["text"] if outputs else "No response."
    else:
        return f"Error: {response.text}"

def deep_plan(task):
    prompt = (
        "You are an expert planning agent. Given a complex task, break it down into clear, actionable subtasks, "
        "and list them as a to-do list."
    )
    return gemini_call(prompt, f"Task to plan: {task}")

def sub_agent(task, context=None):
    prompt = (
        "You are a sub-agent specializing in solving a specific task. "
        "Perform the given task step and generate output that helps complete the overall project."
    )
    return gemini_call(prompt, f"Subtask: {task}", context=context)

# Persistent memory across run
if "plan" not in st.session_state:
    st.session_state["plan"] = []
if "results" not in st.session_state:
    st.session_state["results"] = []

st.title("Deep Agent (Gemini Flash 2.5) — Multipurpose Planner + Subagent Executor")

# Input for user task
user_task = st.text_area("Describe your complex task or project:")

if st.button("Plan Task (Deep Agent)"):
    plan = deep_plan(user_task)
    st.session_state["plan"] = plan.split('\n')
    st.success("Planned! Subtasks created:")

if st.session_state["plan"]:
    for idx, subtask in enumerate(st.session_state["plan"], 1):
        if st.button(f"Execute Subtask {idx}: {subtask}"):
            result = sub_agent(subtask)
            st.session_state["results"].append((subtask, result))

if st.session_state["results"]:
    st.subheader("Sub-Agent Results")
    for subtask, result in st.session_state["results"]:
        st.markdown(f"**{subtask}**: {result}")

st.info(
    "This deep agent app does planning, execution via sub-agents, and keeps results as persistent memory for ultimate end-to-end reasoning."
)
