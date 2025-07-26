#!/bin/bash

log_file="logs/chatbot-$(date +'%Y%m%d-%H%M%S').log"

# ✅ Activate virtual environment
source ./myenv/bin/activate

# 🧼 Clear terminal and print launch info
clear

# 🚀 Start the Streamlit Chatbot App
echo "🚀 Starting Chatbot App at $(date +'%Y-%m-%d %H:%M:%S')."
streamlit run Home.py 2>&1 | tee "$log_file"