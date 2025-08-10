# Home.py

import streamlit as st
from services.ml import load_priority_model, load_sentiment_model
from dotenv import load_dotenv
import joblib
import os

load_dotenv("config/.env")

required_vars = [
    "OPENAI_API_KEY", "GOOGLE_GENAI_API_KEY", "API_MONGO", "EMAIL_ADMIN",
    "EMAIL_PASS", "DOCKER_USERNAME", "DOCKER_PASSWORD", 
    "RENDER_API_KEY", "RENDER_DEPLOY_HOOK",
]
for var in required_vars:
    assert os.getenv(var), f"❌ {var} not defined in .env"

models_dir = "ml/models"
model_files = {
    "priority_pipeline": os.path.join(models_dir, "priority_pipeline.joblib"),
    "sentiment_pipeline": os.path.join(models_dir, "sentiment_pipeline.joblib")
}
loaded_models = {}
for name, path in model_files.items():
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)
    else:
        st.warning(f"⚠️ MoModel not found: {path}")

data_dir = "data"
loaded_data = {}
if os.path.exists(data_dir):
    for file in os.listdir(data_dir):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            try:
                loaded_data[file] = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")

st.set_page_config(page_title="AI Chatbot Suite", page_icon="🤖")
st.title("🤖 Welcome to the AI Chatbot Suite")

st.markdown("""
    This research tool helps IT support teams prioritize service requests using Machine Learning models and chatbot technology.

    Use the navigation below to explore:
""")

st.page_link("pages/Chatbot.py", label="Chat with the AI Assistant", icon="🧠")
st.page_link("pages/Dashboard.py", label="View Dashboard & Analytics", icon="📊")

st.markdown("---")
st.subheader("📌 Project Overview")
st.markdown("""
- Built as part of an MSc Research Project at **Dublin Business School**.
- Combines **BERT** and **GPT** for support ticket triaging.
- Uses **Logistic Regression** and **KMeans** for sentiment & priority classification.
- Technologies: Streamlit, MongoDB, Python, OpenAI, scikit-learn.
""")

st.subheader("🛠️ Maintainer")
st.markdown("""
    **Brenda Lopes** - 20058225@mydbs.ie    
    `MSc Computing & Information Systems`  
    Dublin Business School – 2024|2025
""")

st.markdown("---")
st.info("Use the sidebar menu to navigate between chatbot and dashboard.")

st.markdown("---")
st.markdown("🔗 [GitHub Repository](https://github.com/20058225/chatbot-app)")

st.markdown("""
    <small style='color:gray'>
    Version 1.0.0 – Last updated July 2025  
    Copyright © 2025 Brenda Lopes
    </small>
""", unsafe_allow_html=True)
