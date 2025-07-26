import streamlit as st

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

st.subheader("🛠 Maintainer")
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
