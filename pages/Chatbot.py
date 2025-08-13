# pages/Chatbot.py
# ======================
import streamlit as st
import yagmail
import random
import google.generativeai as genai
import os
from services.mongo import db
from services import ml as ml_services
from datetime import datetime, timezone
from dotenv import load_dotenv
from numpy.linalg import norm
import uuid as _uuid
import pandas as pd
import numpy as np
import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
import logging
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s')
logging.info("🔄️ Reloading Chatbot.py...")
load_dotenv(dotenv_path="config/.env")

try:
    predict_sentiment = ml_services.predict_sentiment
    predict_priority = ml_services.predict_priority
    _ml_available = True
except Exception as e:
    logging.warning("ML services not available: %s", e)
    _ml_available = False

    def predict_sentiment(texts):
        return ["unknown" for _ in texts]
    def predict_priority(texts):
        return ["Low" for _ in texts]

try: 
    genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
except Exception as e:
    logging.warning("Failed to configure genai: %s", e)
    model = None

email_admin = os.getenv("EMAIL_ADMIN")
email_pass = os.getenv("EMAIL_PASS")

users = db["users"]
chats = db["chats"]
faq = db["faq"]
knowledge = db["knowledge"]
unanswered = db["unanswered"]
default_chat = db["default_chat"]
monitoring_col = db["monitoring"]

def log_event(event_type, details, status="success", log_source="production"):
    """Registra um evento no MongoDB (coleção monitoring)."""
    monitoring_col.insert_one({
        "event": event_type,
        "details": details,
        "status": status,
        "log_source": log_source,
        "timestamp": datetime.now(timezone.utc)
    })


@st.cache_data
def load_faq_df():
    return pd.read_csv("data/train_model.csv")


@st.cache_resource(show_spinner=False)
def load_models():
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    return bert_tokenizer, bert_model, gpt2_tokenizer, gpt2_model

bert_tokenizer, bert_model, gpt2_tokenizer, gpt2_model = load_models()


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)


def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze()
    return emb.detach().cpu().numpy()


def generate_gpt2_reply(prompt, max_length=50):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    reply = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply


def predict_sentiment_with_text(text):
    try:
        preds = predict_sentiment([text])
        return str(preds[0])
    except Exception as e:
        logging.error("predict_sentiment error: %s", e)
        return "unknown"


def predict_priority_with_text(text):
    try:
        preds = predict_priority([text])
        return str(preds[0])
    except Exception as e:
        logging.error("predict_priority error: %s", e)
        return "Low"


st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_loaded_for_session" not in st.session_state:
    st.session_state.chat_loaded_for_session = None

def generate_chat_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    short_uid = str(_uuid.uuid4())[:8]
    return f"{timestamp}-{short_uid}"


def is_similar(a, b, threshold=0.85):
    if a is None or b is None:
        return False
    a, b = str(a).lower(), str(b).lower()
    return SequenceMatcher(None, a, b).ratio() > threshold


def get_ai_reply(user_input):
    try:
        faq_entries = list(faq.find({}, {"_id": 0}).limit(10))

        default_doc = default_chat.find_one({"intents": {"$exists": True}}) or {}
        intent_entries = default_doc.get("intents", [])[:5]
        
        knowledge_articles = list(knowledge.find({}, {"_id": 0}).limit(5))
        
        faq_context = "\n".join([f"Q: {e.get('question','')}\nA: {e.get('answer','')}" 
            for e in faq_entries])

        intent_context = "\n".join([
            f"[Intent: {i.get('tag','')}]\n"
            f"Patterns: {', '.join(i.get('patterns', []))}\n"
            f"Responses: {', '.join(i.get('responses', []))}"
            for i in intent_entries
        ])

        kb_context = "\n".join([f"Title: {a.get('title','')}\nContent: {a.get('content','')}"
            for a in knowledge_articles])

        full_context = f"""
You are a support assistant for TechFix Solutions.
Use the company's FAQ, known intent patterns,
and support knowledge base to better answer customer questions.

=== FAQs ===
{faq_context}

=== Intents ===
{intent_context}

=== Knowledge Articles ===
{kb_context}

USER: {user_input}
Respond professionally and clearly based on the context above.
""".strip()

        chat = model.start_chat(history=[])
        response = chat.send_message(full_context)
        return response.text.strip()

    except Exception as e:
        logging.error(f"get_ai_reply error: {e}")

        try:
            return generate_gpt2_reply(user_input)
        except Exception:
            return f"Sorry, there was an error with the AI: {e}"


def generate_bot_response(user_input):
    logging.info("👁️ Checking Patterns...")
    answer, tag = find_default_answer(user_input)
    if answer:
        log_event("chat_response", {
            "user_input": user_input,
            "predicted_source": "intent",
            "predicted_tag": tag
        }, log_source="production")
        return answer, tag or "intent"

    logging.info("👁️ Checking FAQ...")
    answer = find_known_answer(user_input)
    if answer:
        log_event("chat_response", {
            "user_input": user_input,
            "predicted_source": "faq"
        }, log_source="production")
        return answer, "faq"

    logging.info("👁️ Checking Knowledge Base...")
    answer = find_knowledge_answer(user_input)
    if answer:
        log_event("chat_response", {
            "user_input": user_input,
            "predicted_source": "kb"
        }, log_source="production")
        return answer, "kb"

    logging.info("🤖 Calling AI model...")
    answer = get_ai_reply(user_input)
    log_event("chat_response", {
        "user_input": user_input,
        "predicted_source": "ai"
    }, log_source="production")
    return answer, "ai"


def find_default_answer(user_input):
    user_input = user_input.strip().lower()
    doc = default_chat.find_one({"intents": {"$exists": True}}) or {}
    intents = doc.get("intents", [])
    if not "intents":
        return None, None

    for intent in intents:
        for pattern in intent.get("patterns", []):
            if is_similar(pattern, user_input):
                selected_response = random.choice(intent.get("responses", [])) if intent.get("reponse") else None
                return selected_response, intent.get("tag", "intent")
    return None, None


def find_known_answer(user_input):
    user_input = user_input.strip()
    if len(user_input) < 2:
        return None
    
    safe = re.escape(user_input)
    doc = faq.find_one(
        {"question": {"$regex": safe, "$options": "i"}})
    logging.info(f"find_known_answer:{doc}")
    if doc and doc.get("answer"):
        return doc["answer"]

    for f in faq.find({}, {"question": 1, "answer": 1}).limit(50):
        q = f.get("question", "")
        if SequenceMatcher(None, q.lower(), user_input.lower()).ratio() > 0.75:
            return f.get("answer")
    return None


def find_knowledge_answer(user_input):
    if len(user_input.strip()) < 4:
        return None
    safe = re.escape(user_input)
    doc = knowledge.find_one(
        {"$or": [
            {"title": {"$regex": safe, "$options": "i"}}, 
            {"content": {"$regex": safe, "$options": "i"}},
        ]
    })
    return doc.get("content") if doc else None


def save_chat_message(
        user,
        question,
        answer,
        tag=None,
        sentiment=None,
        priority=None,
        embedding=None,
        user_time=None,
        bot_time=None,
        thumbs_up=False,
        thumbs_down=False):
    
    session_id = st.session_state.session_id
    timestamp = datetime.now(timezone.utc)

    if embedding is not None:
        try:
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            else:
                embedding = list(embedding)
        except Exception:
            try:
                embedding = np.array(embedding).tolist()
            except Exception:
                embedding = None
    
    if not session_id:
        session_id = generate_chat_id()
        st.session_state.session_id = session_id

        new_session = {
                "session_id": session_id,
                "user_id": user.get("email") if user else None,
                "start_time": timestamp,
                "messages": [],
                "last_updated": timestamp
            }
        try:
            chats.insert_one(new_session)
        except Exception as e:
            logging.error("Failed to create new session: %s", e)
            return None, None, None

    message_id = str(_uuid.uuid4())
    message = {
        "message_id": message_id,
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
        "user_time": user_time or timestamp,
        "bot_time": bot_time or timestamp,
        "intent_tag": tag,
        "sentiment": sentiment,
        "priority": priority,
        "bert_embedding": embedding,
        "thumbs_up": bool(thumbs_up),
        "thumbs_down": bool(thumbs_down)
    }

    try:
        chats.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message},
                "$set": {"last_updated": timestamp}
            },
            upsert=True
        )
        return session_id, message_id, message
    except Exception as e:
        logging.error(f"❌ Failed to log conversation: {e}")
        return None, None, None
    

def send_email(to, subject, body):
    try:
        yag = yagmail.SMTP(email_admin, email_pass)
        yag.send(to=to, subject=subject, contents=body)
        logging.info(f"✅ Email sent to {to}")
    except Exception as e:
        logging.error(f"❌ Email failed: {e}")
        st.error(f"❌ Failed to send email notification:\n\n`{e}`")


def handle_unanswered(user, question, request_type="unknown"):
    try:
        unanswered.insert_one({
            "user_id": user["email"],
            "question": question,
            "request_type": request_type,
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception as e:
        logging.error("❌ Failed to log unanswered question:", e)
        st.error("❌ Failed to store unanswered question.")

    subject = (f"[Chatbot - Unanswered] New question from {user['email']}")
    body = (
        f"Question: {question}\n"
        f"Type: {request_type}\n"
        f"User: {user['name']} ({user['email']})"
    )
    send_email(email_admin, subject, body)


def handle_feedback(user, question, answer, ticket_id, feedback_text, liked=False):
    feedback_doc = {
        "ticket_id": ticket_id,
        "feedback": feedback_text,
        "timestamp": datetime.now(timezone.utc)
    }
    try:
        db.feedback.insert_one(feedback_doc)
    except Exception as e:
        logging.error("Failed to insert feedback: %s", e)

    thumbs = "👍" if liked else "👎"
    subject = (f"[Chatbot - Feedback] {thumbs} from {user['email']} | {ticket_id}")
    body = (
        f"User: {user['name']} ({user['email']})\n"
        f"{'Liked' if liked else 'Disliked'} Answer\n\n"
        f"Q: {question}\nA: {answer}"
    )
    send_email(email_admin, subject, body)


def get_chat_topic(messages):
    from collections import Counter
    tags = [msg.get("intent_tag") for msg in messages if msg.get("intent_tag")]
    if tags:
        return Counter(tags).most_common(1)[0][0]
    if messages:
        first_question = messages[0].get("question", "")
        if not isinstance(first_question, str):
            first_question = str(first_question)
        keywords = first_question.lower().split()[:3]
        return ", ".join(keywords)
    return "No topic"


def reload_chat_history(force=False):
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
    if st.session_state.get("chat_loaded_for_session") == session_id and not force:
        return
    doc = chats.find_one({"session_id": session_id})
    if doc:
        st.session_state.chat_history = doc.get("messages", [])
    else:
        st.session_state.chat_history = []
    st.session_state["chat_loaded_for_session"] = session_id


def register_form():
    with st.form("register_form"):
        st.subheader("🔐 Start Chat")

        email = st.text_input("Your Email")
        name = st.text_input("Your Name")
        submitted = st.form_submit_button("Start Chat")

        if submitted:
            user = users.find_one({"email": email})
            if not user:
                user = {
                    "name": name,
                    "email": email,
                    "first_seen": datetime.now(timezone.utc),
                    "last_active": datetime.now(timezone.utc)
                }
                users.insert_one(user)
            else:
                users.update_one({"email": email}, {
                    "$set": {"last_active": datetime.now(timezone.utc)}})

            st.session_state.user = user
            logging.info("✅ User registered. Starting chat...")
            st.success("✅ User registered. Starting chat...")

            st.session_state.chat_history = []
            st.session_state.session_id = None
            st.session_state.chat_start_time = None

            st.rerun()


def user_details():
    user = st.session_state.get("user")

    if st.sidebar.button("🔓 Logout"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.input_processed = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Account Info")
    st.sidebar.markdown(f"**Name:** {user['name']}")
    st.sidebar.markdown(f"**Email:** {user['email']}")

    first_seen = user.get("first_seen")
    last_active = user.get("last_active")

    def fmt_date(dt):
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d")
        return str(dt)

    if first_seen:
        st.sidebar.markdown(f"**First Access:** {fmt_date(first_seen)}")
    if last_active:
        st.sidebar.markdown(f"**Last Active:** {fmt_date(last_active)}")

    user_email = user.get("email")
    past_chats = list(chats.find({"user_id": user_email}).sort("start_time", -1))
    st.sidebar.markdown("---")

    def load_chat(session_id):
        chat_data = chats.find_one({"session_id": session_id})
        if chat_data:
            st.session_state.session_id = chat_data.get("session_id")
            st.session_state.chat_start_time = chat_data.get("start_time")
            st.session_state.chat_history = chat_data.get("messages", [])
            st.session_state.chat_loaded_for_session = session_id
            st.success(f"✅ Loaded chat session {session_id}!")

    with st.sidebar.expander("🧾 Previous Chats"):
        for idx, msg in enumerate(past_chats, start=1):
            session_id = msg.get("session_id")
            start_time = msg.get("start_time")
            messages = msg.get("messages", [])
            topic = get_chat_topic(messages)

            # Garantir que start_time é datetime
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except:
                    start_time = None

            display_time = start_time.strftime("%Y-%m-%d %H:%M") if isinstance(start_time, datetime) else "Unknown date"

            # Botão: uso de label simples e dados extras como legenda
            if st.button(f"📂 {idx} | {topic}", key=f"load_{session_id}_{idx}"):
                load_chat(session_id)

            st.caption(f"🕒 {display_time}")


    if st.session_state.get("chat_loaded_success"):
        st.success(f"✅ Chat loaded successfully!")
        logging.info(f"✅ Chat loaded successfully!")
        st.session_state.chat_loaded_success = False


def chat_interface():
    st.title("💬 IT Support Chatbot")

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_user_input", "")

    if st.session_state.get("session_id"):
        reload_chat_history()

    ticket_id = st.session_state.get("session_id")
    if ticket_id:
        st.info(f"🎟️ Your Ticket ID: `{ticket_id}`.")

    # --- Renderiza histórico ---
    for idx, msg in enumerate(st.session_state.chat_history):
        user_time_str = msg.get("user_time")
        try:
            user_time_str = user_time_str.strftime("%H:%M:%S") if isinstance(user_time_str, datetime) else str(user_time_str)
        except Exception:
            user_time_str = str(user_time_str)
        st.markdown(f"👩‍💻 **You** ({user_time_str}): {msg.get('question')}")

        if msg.get("answer"):
            bot_time = msg.get("bot_time")
            try:
                bot_time_str = bot_time.strftime("%H:%M:%S") if isinstance(bot_time, datetime) else str(bot_time)
            except Exception:
                bot_time_str = str(bot_time)
            st.markdown(f"🤖 **Bot** ({bot_time_str}): {msg.get('answer')}")

            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("👍", key=f"thumbsup_{idx}"):
                    handle_feedback(st.session_state.user, msg["question"], msg["answer"],
                                    st.session_state.get("session_id"), feedback_text="like", liked=True)
            with col2:
                if st.button("👎", key=f"thumbsdown_{idx}"):
                    handle_feedback(st.session_state.user, msg["question"], msg["answer"],
                                    st.session_state.get("session_id"), feedback_text="dislike", liked=False)
        else:
            elapsed = 0
            try:
                elapsed = (datetime.now(timezone.utc) - msg["user_time"]).total_seconds()
            except Exception:
                elapsed = 0
            st.markdown("🤖 **Bot**: ⏳ thinking..." if elapsed < 5 else "🤖 **Bot**: 🔍 still looking...")

    # --- Input do usuário ---
    user_input = st.chat_input("Hello! How can I help you today?", key="user_input")

    if user_input and user_input != st.session_state.last_user_input:
        st.session_state.last_user_input = user_input

        if not st.session_state.get("session_id"):
            st.session_state.session_id = generate_chat_id()
            st.session_state.chat_start_time = datetime.now(timezone.utc)

        chats.update_one(
            {"session_id": st.session_state.session_id},
            {"$setOnInsert": {
                "user_id": st.session_state.user.get("email") if st.session_state.user else None,
                "start_time": st.session_state.chat_start_time,
                "messages": []
            }, "$set": {"last_updated": datetime.now(timezone.utc)}},
            upsert=True
        )

        sentiment_pred = predict_sentiment_with_text(user_input)
        priority_pred = predict_priority_with_text(user_input)
        user_time = datetime.now(timezone.utc)

        session_id_ret, message_id, message = save_chat_message(
            st.session_state.user,
            user_input,
            answer=None,
            tag=None,
            sentiment=sentiment_pred,
            priority=priority_pred,
            embedding=None,
            user_time=user_time,
            bot_time=None
        )

        if message is not None:
            st.session_state.chat_history.append(message)
        else:
            st.session_state.chat_history.append({
                "message_id": None,
                "question": user_input,
                "answer": None,
                "user_time": user_time
            })
        st.rerun()


    for idx, msg in enumerate(st.session_state.chat_history):
        if not msg.get("answer"):
            question = msg.get("question")
            answer, tag = generate_bot_response(question)
            bot_time = datetime.now(timezone.utc)            
            emb = get_bert_embeddings(msg["question"])

            try:
                emb = get_bert_embeddings(question)
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            except Exception:
                emb_list = None

            st.session_state.chat_history[idx].update({
                "answer": answer,
                "bot_time": bot_time,
                "intent_tag": tag,
                "bert_embedding": emb_list
            })

            session_id = st.session_state.get("session_id")
            message_id = msg.get("message_id")
            try:
                if message_id:
                    chats.update_one(
                        {"session_id": session_id, "messages.message_id": message_id},
                        {"$set": {
                            "messages.$.answer": answer,
                            "messages.$.bot_time": bot_time,
                            "messages.$.intent_tag": tag,
                            "messages.$.bert_embedding": emb_list
                        }}
                    )
                else:
                    chats.update_one(
                        {"session_id": session_id},
                        {"$push": {"messages": {
                            "message_id": str(_uuid.uuid4()),
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now(timezone.utc),
                            "user_time": msg.get("user_time"),
                            "bot_time": bot_time,
                            "intent_tag": tag,
                            "sentiment": msg.get("sentiment"),
                            "priority": msg.get("priority"),
                            "bert_embedding": emb_list
                        }}}
                    )
            except Exception as e:
                logging.error("Failed to persist bot answer: %s", e)

            st.rerun()
            break

    if st.button("❌ Finish Chat"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.last_user_input = ""
        st.session_state.session_id = None
        st.session_state.chat_start_time = None
        st.rerun()


if "user" not in st.session_state or st.session_state.user is None:
    st.subheader("🔐 Start or Resume a Chat")
    login_mode = st.radio("How would you like to resume your chat?", [
                          "🆕 Start New Chat", "📧 Email", "🎟️ Ticket ID"])

    if login_mode == "📧 Email":
        email = st.text_input("Your Email")

        if st.button("Login via Email"):
            user = users.find_one({"email": email})
            
            if user:
                users.update_one(
                    {"email": email}, 
                    {"$set": {"last_active": datetime.now(timezone.utc)}})
                
                st.session_state.user = user
                last_chat = chats.find_one({"user_id": email}, sort=[("start_time", -1)])
                if last_chat:
                    st.session_state.session_id = last_chat["session_id"]
                    st.session_state.chat_start_time = last_chat.get("start_time")
                    st.session_state.chat_history = last_chat.get("messages", [])
                    st.session_state.chat_loaded_for_session = st.session_state.session_id
                else:
                    st.session_state.session_id = None
                    st.session_state.chat_start_time = None
                    st.session_state.chat_history = []

                logging.info("✅ Logged in!")
                st.success("✅ Logged in!")
                st.rerun()
            else:
                logging.warning("❌ Email not found. Please register.")
                st.warning("❌ Email not found. Please register.")

    elif login_mode == "🎟️ Ticket ID":
        ticket_id = st.text_input("Enter your Ticket ID")

        if st.button("Resume via Ticket ID"):
            chat_data = chats.find_one({"session_id": ticket_id})

            if chat_data:
                email = chat_data["user_id"]
                user = users.find_one({"email": email})

                if user:
                    st.session_state.user = user
                    st.session_state.session_id = ticket_id
                    st.session_state.chat_start_time = chat_data.get(
                        "start_time", datetime.now(timezone.utc))
                    st.session_state.chat_history = chat_data.get("messages", [])
                    st.session_state.chat_loaded_for_session = ticket_id

                    logging.info("✅ Session restored!")
                    st.success("✅ Session restored!")
                    st.rerun()
                else:
                    logging.error("❌ User for this ticket not found.")
                    st.error("❌ User for this ticket not found.")

            else:
                logging.error("❌ Invalid Ticket ID")
                st.error("❌ Invalid Ticket ID")

    elif login_mode == "🆕 Start New Chat":
        register_form()
    else:
        register_form()
else:
    user_details()
    chat_interface()