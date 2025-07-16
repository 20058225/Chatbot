import streamlit as st

st.set_page_config(page_title="AI Chatbot Suite", page_icon="🤖")
st.title("🤖 AI Chatbot Suite")

st.page_link("app/chatbot.py", label="Chat", icon="🧠")
st.page_link("app/dashboard.py", label="Dashboard", icon="📊")