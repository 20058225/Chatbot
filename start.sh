#!/bin/bash

# ✅ Activate your virtual environment
source ./myenv/bin/activate

# ⛔ Exit if venv activation fails
if [[ $? -ne 0 ]]; then
  echo "❌ Failed to activate virtual environment. Make sure './myenv/' exists."
  exit 1
fi

# 🧹 Clean up terminal
clear

echo "🔍 Listing installed packages (before sync):"
#pip list

# ✅ Sync project packages for current environment
echo "📦 Installing dependencies from requirements.txt..."
#pip freeze > requirements.txt
#pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
  echo "✅ All requirements installed successfully!"
else
  echo "❌ Failed to install packages from requirements.txt"
  exit 1
fi

# 🧽 Clean terminal again before app launch
clear

# 🧠 Launch the Streamlit chatbot
echo "🚀 Starting Chatbot App..."
#streamlit run Home.py
streamlit run Home.py 2>&1 | tee chatbot-app.log
