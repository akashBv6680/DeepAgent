import streamlit as st
import requests
import json
import time

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
API_KEY = st.secrets["GEMINI_API_KEY"]

def gemini_call(system_prompt, user_query, context=None, max_retries=5, backoff=5):
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": user_query})
    data = {"contents": [{"parts": [{"text": json.dumps(messages)}]}]}
    url = f"{API_URL}?key={API_KEY}"

    last_error = None
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=data)
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
    return f"Error: Gemini API is overloaded after {max_retries} attempts. Please wait and try again."

def deep_plan(task):
    prompt = (
        "You are an expert planning agent. Break this into clear, actionable steps (to-do list)."
    )
    return gemini_call(prompt, f"Task: {task}")

def sub_agent(task, context=None):
    prompt = (
        "You are a sub-agent. Solve your assigned task step and output the result with minimal payload."
    )
    return gemini_call(prompt, f"Subtask: {task}", context=context)

if "plan" not in st.session_state:
    st.session_state["plan"] = []
if "results" not in st.session_state:
    st.session_state["results"] = []
if "subtask_busy" not in st.session_state:
    st.session_state["subtask_busy"] = False

st.title("Deep Agent — Reliable Planner & Executor")

user_task = st.text_area("Describe your complex task or project:")

if st.button("Plan Task (Deep Agent)", disabled=not user_task.strip()):
    plan = deep_plan(user_task)
    st.session_state["plan"] = [line.strip() for line in plan.split('\n') if line.strip()]
    st.session_state["results"] = []
    st.session_state["subtask_busy"] = False
    st.success("Planned! Subtasks created.")

if st.session_state["plan"]:
    st.subheader("Planned Subtasks")
    for idx, subtask in enumerate(st.session_state["plan"], 1):
        if st.session_state["subtask_busy"]:
            st.info("A subtask is still running or API is busy. Please wait before starting another.")
            break
        if st.button(f"Execute Subtask {idx}: {subtask}", key=f"subtask_{idx}"):
            st.session_state["subtask_busy"] = True
            result = sub_agent(subtask)
            if "overloaded" in result or "UNAVAILABLE" in result:
                st.warning("Gemini API overloaded/busy. Wait 1-2 minutes and retry this subtask.")
            elif "Error" in result:
                st.error(result)
            else:
                st.session_state["results"].append((subtask, result))
                st.success(f"{subtask}: Completed!")
            st.session_state["subtask_busy"] = False

if st.session_state["results"]:
    st.subheader("Sub-Agent Results")
    for subtask, result in st.session_state["results"]:
        st.markdown(f"**{subtask}**: {result}")

st.info(
    "Gemini free API can be overloaded. Space out task attempts and keep prompts compact for highest success."
)
