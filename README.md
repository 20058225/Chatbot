# 🤖 AI Chatbot Suite

A Streamlit-based AI Chatbot application with support for sentiment and priority classification of helpdesk tickets, FAQ import, and MongoDB integration. Includes training pipelines using machine learning models stored locally.

---

## 🚀 Features

- 📥 Import FAQs, default messages, or knowledge articles via CSV, JSON, or manual form.
- 🧠 Chatbot interface powered by OpenAI (via API key).
- 🧪 Sentiment & priority detection using pre-trained ML models (`joblib` pipelines).
- 📊 Admin dashboard for chat insights.
- 🐳 Docker & GitHub Actions integration for CI/CD.

---

## 🧰 Technologies

- Python 3.11
- Streamlit
- MongoDB (via `pymongo`)
- scikit-learn, joblib
- Docker
- GitHub Actions

---

## 📂 Project Structure
```
.
├── ml/
│ └── models/
│ ├── sentiment_pipeline.joblib
│ └── priority_pipeline.joblib
├── data/
│ └── tickets.csv
├── pages/
│ ├── Chatbot.py
│ └── Dashboard.py
├── services/
│ └── utils.py
├── config/
│ └── .env
├── Home.py
├── start.sh
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

---

## ⚙️ Setup Instructions

### 🔧 Local (with Virtual Environment)

```bash
# Clone repo and move into folder
git clone https://github.com/20058225/chatbot-app.git
cd chatbot-app

# Create venv and activate it
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
```
Or use:
```
bash start.sh
```

---

## 🐳 Run with Docker
```
# Build Docker image
docker build -t chatbot-app .

# Run container
docker run -p 8501:8501 --env-file=config/.env chatbot-app

```

---

## 🔐 Environment Variables (.env)
Place this file in config/.env:
```
API_MONGO=mongodb+srv://<your-connection>
OPENAI_API_KEY=sk-...
EMAIL_ADMIN=your@email.com
EMAIL_PASS=your-app-password
```

---

## ✅ Run Tests
```
pytest tests/
```

---

## 🧠 ML Models
```
ml/models/sentiment_pipeline.joblib: Sentiment classifier (positive/neutral/negative)

ml/models/priority_pipeline.joblib: Ticket priority classifier (High/Medium/Low)

Both loaded at startup by services/utils.py.
```

---

## 🗃️ Sample Data Format (data/tickets.csv)

This dataset is used to train and evaluate both ML pipelines:

- **Sentiment classification** → `positive`, `neutral`, `negative`
- **Priority classification** → `High`, `Medium`, `Low`

### 🔍 Example rows:

```csv
description,sentiment,priority
"I can't log in to my account",negative,High
"My computer is running slow",negative,Medium
"How do I reset my email password?",neutral,Medium
"The printer is showing a paper jam error",negative,Medium
"Request for software upgrade",neutral,Low
"Thank you for fixing my internet issue",positive,Low
```
---

## 🔄 CI/CD with GitHub Actions
- Triggers on push to main

- Installs dependencies and runs pytest

- Builds Docker image for deployment

See .github/workflows/chatbot-app.yml.

---

## 📄 License
MIT © 2025 Brenda Lopes — [LICENSE](./LICENSE)

---

## ✨ Screenshots

### 🤖 Chatbot Interface
![Chatbot UI](assets/chatbot.png)

### 📊 Admin Dashboard
![Dashboard](assets/dashboard.png)

### 📥 FAQ & Default Message Import
![Import Interface](assets/faqs.png)
---

## 🙋‍♀️ About
This project was developed as part of the MSc in Computing & Information Systems at Dublin Business School. The goal is to improve ticket triage using NLP, ML and a chatbot interface.

```

```


