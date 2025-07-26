#!/bin/bash

# 🧼 Clear terminal
clear

log_file="logs/chatbot-$(date +'%Y%m%d-%H%M%S').log"

# ✅ Activate virtual environment
source ./myenv/bin/activate

if [[ $? -ne 0 ]]; then
  echo "❌ Failed to activate virtual environment. Make sure './myenv/' exists."
  exit 1
fi

# 📦 Auto-generate requirements.txt if missing
if [[ ! -f requirements.txt ]]; then
  echo "📄 Generating requirements.txt..."
  pip freeze > requirements.txt
fi

# 📦 Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
# pip install -r requirements.txt
pip install --no-deps --requirement requirements.txt --upgrade --quiet

if [[ $? -ne 0 ]]; then
  echo "❌ Failed to install packages"
  exit 1
fi

# 🧼 Clear terminal and print launch info
clear

# 🚀 Start the Streamlit Chatbot App
#sh ./start.sh

echo "🚀 Starting Chatbot App at ⌚$(date +'%Y-%m-%d %H:%M:%S')."
streamlit run Home.py 2>&1 | tee "$log_file"