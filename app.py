import streamlit as st
import requests
import json
import time
import datetime

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
API_KEY = st.secrets["GEMINI_API_KEY"]

# Retry and latency-aware Gemini call
def gemini_call(system_prompt, user_query, context=None, max_retries=3, backoff=2):
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": user_query})
    data = {"contents": [{"parts": [{"text": json.dumps(messages)}]}]}
    url = f"{API_URL}?key={API_KEY}"

    last_error = None
    for attempt in range(max_retries):
        start_time = datetime.datetime.now()
        response = requests.post(url, headers=headers, json=data)
        end_time = datetime.datetime.now()
        latency = (end_time - start_time).total_seconds()
        st.session_state["timing"].append(latency)
        
        if response.status_code == 200:
            outputs = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            if outputs:
                return outputs[0]["text"]
            else:
                last_error = "No response."
        elif response.status_code == 503 or "overloaded" in response.text or "UNAVAILABLE" in response.text:
            last_error = "API overloaded or unavailable. Retrying..."
            time.sleep(backoff * (attempt + 1))
        else:
            last_error = response.text
            break
    return f"Error: {last_error}"

def deep_plan(task):
    prompt = (
        "You are an expert planning agent. Given a complex task, break it down into clear, actionable subtasks, "
        "and list them as a numbered to-do list."
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
if "timing" not in st.session_state:
    st.session_state["timing"] = []

st.title("Deep Agent (Gemini Flash 2.5) — Multipurpose Planner + Subagent Executor")

# Sidebar: Agent performance metrics
if st.session_state["timing"]:
    avg_latency = sum(st.session_state["timing"]) / len(st.session_state["timing"])
    st.sidebar.metric("Avg Latency (sec)", f"{avg_latency:.2f}")
    st.sidebar.metric("Agent Calls", len(st.session_state["timing"]))

# Input for user task
user_task = st.text_area("Describe your complex task or project:")

if st.button("Plan Task (Deep Agent)", disabled=not user_task.strip()):
    plan = deep_plan(user_task)
    st.session_state["plan"] = [line.strip() for line in plan.split('\n') if line.strip()]
    st.session_state["results"] = []
    st.success("Planned! Subtasks created:")

if st.session_state["plan"]:
    st.subheader("Planned Subtasks")
    for idx, subtask in enumerate(st.session_state["plan"], 1):
        col1, col2 = st.columns((5,1))
        # Only allow one execution at a time to avoid API overload
        with col1:
            if st.button(f"Execute Subtask {idx}: {subtask}", key=f"subtask_{idx}"):
                result = sub_agent(subtask)
                # Friendly error handling
                if ("Error" in result or 
                    "UNAVAILABLE" in result or 
                    "overloaded" in result or "503" in result):
                    st.warning("Gemini API is currently overloaded or unavailable. Please try again later.")
                else:
                    st.session_state["results"].append((subtask, result))
                    st.success(f"{subtask}: Completed!")
        with col2:
            # Show status only for the finished tasks
            if idx <= len(st.session_state["results"]):
                st.caption("✅")

if st.session_state["results"]:
    st.subheader("Sub-Agent Results")
    for subtask, result in st.session_state["results"]:
        st.markdown(f"**{subtask}**: {result}")

st.info(
    "This deep agent app plans complex tasks, handles sub-agent execution with error recovery and displays performance metrics (latency, API health) in the sidebar."
)
