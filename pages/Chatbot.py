import streamlit as st
import yagmail
import random
import google.generativeai as genai
import os
from services.mongo import db
from datetime import datetime, timezone
from dotenv import load_dotenv
from uuid import uuid4
import joblib
from difflib import SequenceMatcher
from ml.sentiment import MODEL_PATH as SENT_MODEL
from ml.priority import MODEL_PATH as PRIO_MODEL

load_dotenv(dotenv_path="config/.env")

_sentiment_model = joblib.load(SENT_MODEL)
_priority_model = joblib.load(PRIO_MODEL)

genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

email_admin = os.getenv("EMAIL_ADMIN")
email_pass = os.getenv("EMAIL_PASS")

users = db["users"]
chats = db["chats"]
faq = db["faq"]
unanswered = db["unanswered"]
default_chat = db["default_chat"]

def is_similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.85

def get_ai_reply(user_input):
    try:
        # Gather contextual data to send with prompt
        faq_entries = list(faq.find({}, {"_id": 0}).limit(10))
        intent_entries = list(default_chat.find_one({"intents": {"$exists": True}})["intents"])[:5]
        knowledge_articles = list(db["knowledge"].find({}, {"_id": 0}).limit(5))


        # Convert into context strings
        faq_context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in faq_entries])
        intent_context = "\n".join([f"[Intent: {i['tag']}]\nPatterns: {', '.join(i.get('patterns', []))}\nResponses: {', '.join(i.get('responses', []))}" for i in intent_entries])
        kb_context = "\n".join([f"Title: {a['title']}\nContent: {a['content']}" for a in knowledge_articles])

        full_context = f"""
            You are a support assistant for TechFix Solutions. Use the company’s FAQ, known intent patterns, and support knowledge base to better answer customer questions.

            === FAQs ===
            {faq_context}

            === Intents ===
            {intent_context}

            === Knowledge Articles ===
            {kb_context}

            USER: {user_input}
            Respond professionally and clearly based on the context above.
        """.strip()

        # Send message with context in prompt
        chat = model.start_chat(history=[])
        response = chat.send_message(full_context)
        return response.text.strip()

    except Exception as e:
        return f"Sorry, there was an error with the AI: {e}"

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🧠 Intelligent Support Chatbot")

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    

def predict_sentiment(text):
    return _sentiment_model.predict([text])[0]

def predict_priority(text):
    return _priority_model.predict([text])[0]

def find_default_answer(response_type):
    doc = default_chat.find_one({"intents": {"%exists": True}})
    if not doc:
        return "I don't understand that."
    for intent in doc["intents"]:
        if intent["tag"].lower() == response_type.lower():
            return random.choice(intent.get("responses", ["I don't understand that."]))
    return "I don't understand that."

def find_known_answer(user_input):
    print(f"User input: {user_input}")
    doc = faq.find_one(
        {"question": {"$regex": user_input, "$options": "i"}}
    )
    print(f"find_known_answer:{doc}")
    return doc["answer"] if doc else None

def find_knowledge_answer(user_input):
    doc = db["knowledge"].find_one({"$or": [
        {"title": {"$regex": user_input, "$options": "i"}},
        {"content": {"$regex": user_input, "$options": "i"}}
    ]})
    print(f"find_knowledge_answer:{doc}")
    if doc:
        return doc.get("content")
    return None

def find_intent_answer(user_input):
    user_input = user_input.strip().lower()
    doc = default_chat.find_one({"intents": {"$exists": True}})
    print(f"find_intent_answer:{doc}") 

    if not doc or "intents" not in doc:
        return None, None
    
    for intent in doc["intents"]:
        for pattern in intent.get("patterns", []):
            #if pattern.lower() in user_input.lower():
            if is_similar(pattern, user_input):
                selected_response = random.choice(intent.get("responses", []))
                return selected_response, intent["tag"]
    return None, None

def generate_bot_response(user_input):
    print("👁️ Checking FAQ...")
    answer = find_known_answer(user_input)
    if answer:
        print("✅ Matched FAQ.")
        return answer

    print("👁️ Checking Knowledge Base...")
    answer = find_knowledge_answer(user_input)
    if answer:
        print("✅ Matched Knowledge.")

        return answer

    print("👁️ Checking Patterns...")
    answer = find_intent_answer(user_input)
    if answer:
        print("✅ Matched Intent.")

        return answer

    print("🤖 Calling AI model...")
    answer = get_ai_reply(user_input)

    if not answer:
        print("⚠️ Nothing worked — using fallback.")
        answer = find_default_answer("fallback")
        handle_unanswered(st.session_state.user, user_input, st.session_state.req_type)

    return answer


def send_email(to, subject, body):
    try:
        yag = yagmail.SMTP(email_admin, email_pass)
        yag.send(to=to, subject=subject, contents=body)
        print(f"✅ Email sent to {to}")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"❌ Email failed: {error_msg}")  # aparece no terminal
        st.warning(f"⚠️ Failed to send email notification:\n\n`{error_msg}`")

def handle_unanswered(user, question, request_type="unknown"):
    try:
        unanswered.insert_one({
            "user_id": user["email"],
            "question": question,
            "request_type": request_type,
            "timestamp": datetime.now(timezone.utc)
        })
    except Exception as e:
        print("❌ Failed to log unanswered question:", e)
        st.warning("⚠️ Failed to store unanswered question.")

    # Notify admin
    subject = f"[Chatbot - Unanswered] New question from {user['email']}"
    body = f"Question: {question}\nType: {request_type}\nUser: {user['name']} ({user['email']})"
    send_email(email_admin, subject, body)

def handle_feedback(user, question, answer, liked=False):
    thumbs = "👍" if liked else "👎"
    subject = f"[Chatbot - Feedback] {thumbs} from {user['email']}"
    body = f"User: {user['name']} ({user['email']})\n{'Liked' if liked else 'Disliked'} Answer\n\nQ: {question}\nA: {answer}"    
    send_email(email_admin, subject, body)

def log_conversation(user, question, answer, sentiment=None, priority=None, thumbs_up=False, thumbs_down=False):
    session_id = st.session_state.session_id
    timestamp = datetime.now(timezone.utc)

    message = {
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
        "sentiment": sentiment,
        "priority": priority,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down
    }

    try:
        chats.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": message
                },
                "$set": {
                    "last_updated": timestamp
                }
            },
            upsert = True
        )
    except Exception as e:
        print("❌ Failed to log conversation:", e)

def register_form():
    with st.form("register_form"):
        st.subheader("🔐 Start Chat")
        email = st.text_input("Your Email")
        name = st.text_input("Your Name")
        req_type = st.selectbox("What do you need help with?", ["Issue", "FAQ", "Other"])
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
                users.update_one({"email": email}, {"$set": {"last_active": datetime.now(timezone.utc)}})

            st.session_state.user = user
            st.session_state.req_type = req_type
            st.success("You're logged in!")

            if "user" not in st.session_state:
                st.session_state.user = None
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid4())
            if "chat_start_time" not in st.session_state:
                st.session_state.chat_start_time = datetime.now(timezone.utc)
            
            chats.insert_one({
                "session_id": st.session_state.session_id,
                "user_id": email,
                "start_time": datetime.now(timezone.utc),
                "messages": []
            })
            
            st.rerun()

def user_details():
    st.sidebar.markdown("### 👤 Account Info")
    st.sidebar.markdown(f"**Name:** {st.session_state.user['name']}")
    st.sidebar.markdown(f"**Email:** {st.session_state.user['email']}")

    first_seen = st.session_state.user.get("first_seen")
    if first_seen:
        st.sidebar.markdown(f"**First Access:** {first_seen.strftime('%Y-%m-%d')}")
    
    last_active = st.session_state.user.get("last_active")
    if last_active:
        st.sidebar.markdown(f"**Last Active:** {last_active.strftime('%Y-%m-%d | %H:%M:%S')}")
    
    if st.sidebar.button("🔓 Logout"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

def chat_interface():
    st.markdown("---")
    st.subheader("🤖 Chat with the Bot")
    st.info(f"🎟️ Your Ticket ID: `{st.session_state.session_id}`\nCopy this to resume the chat later.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
        chats.insert_one({
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user["email"],
            "start_time": datetime.now(timezone.utc),
            "messages": []
        })

    if st.session_state.get("input_processed", False):
        st.session_state.input_processed = False
        st.session_state.user_input = ""

    for i, msg in enumerate(st.session_state.chat_history):
        q_time = msg.get("time")
        time_str = q_time.strftime("%H:%M:%S") if q_time else "now"
        
        user_msg = msg.get("question", "[missing question]")
        bot_msg = msg.get("answer", "[missing answer]")

        st.markdown(f"👩‍💻 **You**: ({time_str}): {user_msg}")
        st.markdown(f"🤖 **Bot**: ({time_str}): {bot_msg}")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("👍", key=f"like_{i}"):
                st.session_state.chat_history[i]["thumbs_up"] = True
                st.session_state.chat_history[i]["thumbs_down"] = False

                chats.update_one(
                    {"session_id": st.session_state.session_id},
                    {"$set": {
                        f"messages.{i}.thumbs_up": True,
                        f"messages.{i}.thumbs_down": False
                    }}
                )
                
                handle_feedback(
                    st.session_state.user,
                    msg.get("question") or msg.get("q", ""),
                    msg.get("answer") or msg.get("a", ""),
                    liked=True
                )
                st.success("Thanks for the feedback!")
                st.rerun()
        with col2:
            if st.button("👎", key=f"dislike_{i}"):
                st.session_state.chat_history[i]["thumbs_down"] = True
                st.session_state.chat_history[i]["thumbs_up"] = False

                chats.update_one(
                    {"session_id": st.session_state.session_id},
                    {"$set": {
                        f"messages.{i}.thumbs_down": True,
                        f"messages.{i}.thumbs_up": False
                    }}
                )
                
                handle_feedback(
                    st.session_state.user,
                    msg.get("question") or msg.get("q", ""),
                    msg.get("answer") or msg.get("a", ""),
                    liked=False
                )
                st.success("Thanks, we’ll work to improve that!")
                st.rerun()
    
    user_input = st.text_input("Type your question and press Enter", key="user_input")
    if user_input and not st.session_state.get("input_processed", False):
        print(f"User input: {user_input}")

        answer = find_known_answer(user_input)
        if answer is None:
            answer = find_intent_answer(user_input)
        if answer is None:
            answer = find_default_answer(user_input)
        if answer is None:
            answer = find_knowledge_answer(user_input)
        if answer is None:
            answer = get_ai_reply(user_input)
        if not answer:
            answer = get_ai_reply("fallback")
            handle_unanswered(st.session_state.user, user_input, st.session_state.req_type)

        sentiment = predict_sentiment(user_input)
        priority = predict_priority(user_input)

        st.session_state.chat_history.append({
            "question": user_input, 
            "answer": answer,         
            "sentiment": sentiment,
            "priority": priority,
            "time": datetime.now(timezone.utc)})

        log_conversation(st.session_state.user, user_input, answer, sentiment, priority)

        st.session_state.input_processed = True
        st.rerun()
    
    if st.button("❌ Finish Chat"):
        st.success("✅ Conversation closed.")
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.greeting_shown = False
        st.rerun()


if not st.session_state.user:
    st.subheader("🔐 Start or Resume a Chat")

    login_mode = st.radio("How would you like to resume your chat?", ["🆕 Start New Chat", "📧 Email", "🎟️ Ticket ID"])

    if login_mode == "📧 Email":
        email = st.text_input("Your Email")
        if st.button("Login via Email"):
            user = users.find_one({"email": email})
            if user:
                users.update_one({"email": email}, {"$set": {"last_active": datetime.now(timezone.utc)}})
                st.session_state.user = user
                past_chats = list(chats.find({"user_id": email}).sort("start_time", -1))

                if past_chats:
                    st.markdown("### 🧾 Your Previous Conversations")
                    selected = st.selectbox(
                        "Select a past conversation to load",
                        [f"{chat['session_id']} | {chat['start_time'].strftime('%Y-%m-%d %H:%M')}" for chat in past_chats],
                        index=0, key="session_selector"
                    )

                    if st.button("📂 Load Selected Chat"):
                        selected_session_id = selected.split(" | ")[0]
                        selected_chat = chats.find_one({"session_id": selected_session_id})

                        if selected_chat:
                            st.session_state.session_id = selected_chat["session_id"]
                            st.session_state.chat_start_time = selected_chat["start_time"]
                            st.session_state.chat_history = selected_chat.get("messages", [])
                            st.success(f"✅ Loaded session: {selected_session_id}")
                            st.rerun()

                st.session_state.req_type = "Login"

                if "session_id" not in st.session_state:
                    st.session_state.session_id = str(uuid4())
                if "chat_start_time" not in st.session_state:
                    st.session_state.chat_start_time = datetime.now(timezone.utc)    
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                chats.insert_one({
                    "session_id": st.session_state.session_id,
                    "user_id": email,
                    "start_time": st.session_state.chat_start_time,
                    "messages": []
                })
                print("✅ Logged in!")
                st.success("✅ Logged in!")
                st.rerun()        
            else:
                st.warning("❌ Email not found. Please register.")
    
    elif login_mode == "🎟️ Ticket ID":
        ticket_id = st.text_input("Enter your Ticket ID")
        if st.button("Resume via Ticket ID"):
            chat_data = chats.find_one({"session_id": ticket_id})
            if chat_data:
                email = chat_data["user_id"]
                user = users.find_one({"email": email})

                st.session_state.user = user
                st.session_state.session_id = ticket_id
                st.session_state.chat_start_time = chat_data.get("start_time", datetime.now(timezone.utc))
                st.session_state.chat_history = chat_data.get("messages", [])
                st.success("✅ Session restored!")
                st.rerun()
            else:
                st.error("❌ Invalid Ticket ID")
    
    elif login_mode == "🆕 Start New Chat":
        register_form()
    
    else:
        register_form()
else:
    user_details()

    if "greeting_shown" not in st.session_state:
        greeting = find_default_answer("greeting")
        if greeting == "I don't understand that.":
            greeting = "Hello! 👋 How can I assist you today?"
        st.info(greeting)
        st.session_state.greeting_shown = True

    chat_interface()