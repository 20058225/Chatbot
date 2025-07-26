import streamlit as st
import yagmail
import random
import google.generativeai as genai
import os
from services.mongo import db
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid
import joblib
import logging
from difflib import SequenceMatcher
from ml.sentiment import MODEL_PATH as SENT_MODEL
from ml.priority import MODEL_PATH as PRIO_MODEL

logging.info("🔧 Recarregando Chatbot.py...")

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
        intent_entries = list(default_chat.find_one(
            {"intents": {"$exists": True}})["intents"])[:5]
        knowledge_articles = list(
            db["knowledge"].find(
                {}, {
                    "_id": 0}).limit(5))

        # Convert into context strings
        faq_context = "\n".join(
            [f"Q: {e['question']}\nA: {e['answer']}" for e in faq_entries]
        )

        intent_context = "\n".join([
            f"[Intent: {i['tag']}]\n"
            f"Patterns: {', '.join(i.get('patterns', []))}\n"
            f"Responses: {', '.join(i.get('responses', []))}"
            for i in intent_entries
        ])

        kb_context = "\n".join(
            [
                f"Title: {a['title']}\nContent: {a['content']}"
                for a in knowledge_articles]
        )

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

        # Send message with context in prompt
        chat = model.start_chat(history=[])
        response = chat.send_message(full_context)
        return response.text.strip()

    except Exception as e:
        return f"Sorry, there was an error with the AI: {e}"


st.set_page_config(page_title="Chatbot", page_icon="🤖")

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def predict_sentiment(text):
    return _sentiment_model.predict([text])[0]


def predict_priority(text):
    return _priority_model.predict([text])[0]


def generate_chat_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    short_uid = str(uuid.uuid4())[:8]
    return f"{timestamp}-{short_uid}"


def find_default_answer(response_type):
    doc = default_chat.find_one({"intents": {"$exists": True}})
    if not doc:
        return "I don't understand that."
    for intent in doc["intents"]:
        if intent["tag"].lower() == response_type.lower():
            return random.choice(
                intent.get(
                    "responses",
                    ["I don't understand that."]))
    return "I don't understand that."


def find_known_answer(user_input):
    if len(user_input.strip()) < 2:
        return None
    doc = faq.find_one(
        {"question": {"$regex": fr"\b{user_input}\b", "$options": "i"}}
    )
    logging.info(f"find_known_answer:{doc}")
    return doc["answer"] if doc else None


def find_knowledge_answer(user_input):
    if len(user_input.strip()) < 4:
        return None
    doc = db["knowledge"].find_one({"$or": [
        {"title": {"$regex": user_input, "$options": "i"}},
        {"content": {"$regex": user_input, "$options": "i"}}
    ]})
    logging.info(f"find_knowledge_answer:{doc}")
    return doc.get("content") if doc else None


def find_intent_answer(user_input):
    user_input = user_input.strip().lower()
    doc = default_chat.find_one({"intents": {"$exists": True}})
    # logging.info(f"find_intent_answer:{doc}")

    if not doc or "intents" not in doc:
        return None, None

    for intent in doc["intents"]:
        for pattern in intent.get("patterns", []):
            # if pattern.lower() in user_input.lower():
            if is_similar(pattern, user_input):
                selected_response = random.choice(intent.get("responses", []))
                return selected_response, intent["tag"]
    return None, None


def generate_bot_response(user_input):
    logging.info("👁️ Checking Patterns...")
    answer, tag = find_intent_answer(user_input)
    if answer:
        logging.info("✅ Matched Intent.")
        return answer, tag

    logging.info("👁️ Checking FAQ...")
    answer = find_known_answer(user_input)
    if answer:
        logging.info("✅ Matched FAQ.")
        return answer

    logging.info("👁️ Checking Knowledge Base...")
    if len(user_input) > 4:
        answer = find_knowledge_answer(user_input)
        if answer:
            logging.info("✅ Matched Knowledge.")
            return answer, None

    logging.info("🤖 Calling AI model...")
    answer = get_ai_reply(user_input)

    # if not answer:
    #   logging.error("⚠️ Nothing worked — using fallback.")
    #  answer = find_default_answer("fallback")
    # handle_unanswered(
    # st.session_state.user, user_input, st.session_state.req_type)

    return answer, "ai"


def send_email(to, subject, body):
    try:
        yag = yagmail.SMTP(email_admin, email_pass)
        yag.send(to=to, subject=subject, contents=body)
        logging.info(f"✅ Email sent to {to}")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logging.error(f"❌ Email failed: {error_msg}")
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
        logging.error("❌ Failed to log unanswered question:", e)
        st.warning("⚠️ Failed to store unanswered question.")

    # Notify admin
    subject = (f"[Chatbot - Unanswered] New question from {user['email']}")
    body = (
        f"Question: {question}\n"
        f"Type: {request_type}\n"
        f"User: {user['name']} ({user['email']})"
    )
    send_email(email_admin, subject, body)


def handle_feedback(user, question, answer, liked=False):
    thumbs = "👍" if liked else "👎"
    subject = (f"[Chatbot - Feedback] {thumbs} from {user['email']}")
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
        common_tag = Counter(tags).most_common(1)[0][0]
        return common_tag
    else:
        if messages:
            first_question = messages[0].get("question", "")
            keywords = first_question.lower().split()[:3]
            return ", ".join(keywords)
    return "No topic"


def log_conversation(
        user,
        question,
        answer,
        tag=None,
        sentiment=None,
        priority=None,
        thumbs_up=False,
        thumbs_down=False):
    session_id = st.session_state.session_id
    timestamp = datetime.now(timezone.utc)

    message = {
        "question": question,
        "answer": answer,
        "timestamp": timestamp,
        "intent_tag": tag,
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
            upsert=True
        )
    except Exception as e:
        logging.error("❌ Failed to log conversation:", e)


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
            # st.session_state.req_type = req_type
            logging.info("✅ You're logged in!")
            st.success("✅ You're logged in!")

            if "user" not in st.session_state:
                st.session_state.user = None
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "session_id" not in st.session_state:
                st.session_state.session_id = generate_chat_id()
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
    if st.sidebar.button("🔓 Logout"):
        st.session_state.user = None
        st.session_state.chat_history = []
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 👤 Account Info")
    st.sidebar.markdown(f"**Name:** {st.session_state.user['name']}")
    st.sidebar.markdown(f"**Email:** {st.session_state.user['email']}")

    first_seen = st.session_state.user.get("first_seen")
    if first_seen:
        st.sidebar.markdown(
            f"**First Access:** {first_seen.strftime('%Y-%m-%d')}")

    last_active = st.session_state.user.get("last_active")
    if last_active:
        st.sidebar.markdown(
            f"**Last Active:** {last_active.strftime('%Y-%m-%d | %H:%M:%S')}")

    user_email = st.session_state.user["email"]
    past_chats = list(chats.find(
        {"user_id": user_email}).sort("start_time", -1))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧾 Previous Chats")
    with st.sidebar.expander("🧾 Previous Chats"):
        for idx, chat in enumerate(past_chats, start=1):
            session_id = chat.get("session_id")
            start_time = chat.get("start_time")
            messages = chat.get("messages", [])

            topic = get_chat_topic(messages)

            display_time = start_time.strftime(
                "%Y-%m-%d %H:%M") if start_time else "Unknown date"

            button_label = f"📂 Load Chat {idx} | {topic} — {display_time}"

            if st.sidebar.button(button_label, key=f"load_{session_id}_{idx}"):
                st.session_state.session_id = session_id
                st.session_state.chat_start_time = start_time
                st.session_state.chat_history = messages
                logging.info(f"✅ Loaded chat session {session_id}!")
                st.success(f"✅ Loaded chat session {session_id}!")
                st.rerun()


def chat_interface():
    st.title("💬 IT Support Chatbot")

    user_input = st.chat_input("Ask a question...", key="user_input")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if (
        "input_processed" not in st.session_state
        or st.session_state.input_processed
    ):
        st.session_state.input_processed = False

    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = generate_chat_id()
        st.session_state.chat_start_time = datetime.now(timezone.utc)
        chats.insert_one({
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user["email"],
            "start_time": st.session_state.chat_start_time,
            "messages": []
        })

    ticket_id = st.session_state.session_id
    st.info(f"🎟️ Your Ticket ID: `{ticket_id}`.")

    if user_input and not st.session_state.input_processed:
        timestamp = datetime.now().strftime("%H:%M:%S")
        response = generate_bot_response(user_input)

        if isinstance(response, tuple):
            answer, tag = response
        else:
            answer = response
            tag = None

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": answer,
            "intent_tag": tag,
            "sentiment": predict_sentiment(user_input),
            "priority": predict_priority(user_input),
            "time": datetime.now(timezone.utc)
        })

        st.markdown(f"👩‍💻 **You** ({timestamp}): {user_input}")
        st.markdown(f"🤖 **Bot** ({timestamp}): {answer}")
        if tag:
            st.caption(f"🧠 Tag: `{tag}`")

        log_conversation(
            st.session_state.user,
            user_input,
            answer,
            tag,
            st.session_state.chat_history[-1]["sentiment"],
            st.session_state.chat_history[-1]["priority"]
        )

        st.session_state.input_processed = True
        st.rerun()

    if st.button("❌ Finish Chat"):
        logging.info("✅ Conversation closed.")
        st.success("✅ Conversation closed.")
        st.session_state.user = None
        st.session_state.chat_history = []
        st.session_state.input_processed = False
        st.session_state.session_id = None
        st.session_state.chat_start_time = None
        st.rerun()


if not st.session_state.user:

    st.subheader("🔐 Start or Resume a Chat")

    login_mode = st.radio("How would you like to resume your chat?", [
                          "🆕 Start New Chat", "📧 Email", "🎟️ Ticket ID"])

    if login_mode == "📧 Email":
        email = st.text_input("Your Email")
        if st.button("Login via Email"):
            user = users.find_one({"email": email})
            if user:
                users.update_one({"email": email}, {
                    "$set": {"last_active": datetime.now(timezone.utc)}})
                st.session_state.user = user
                past_chats = list(chats.find(
                    {"user_id": email}).sort("start_time", -1))

                if past_chats:
                    st.markdown("### 🧾 Your Previous Conversations")

                    options = []
                    for chat in past_chats:
                        start_time_str = chat['start_time'].strftime(
                            '%Y-%m-%d %H:%M'
                        ) if chat.get('start_time') else "Unknown date"
                        topic = get_chat_topic(chat.get("messages", []))
                        option_label = (
                            f"{start_time_str} | {chat['session_id']} | {topic}"
                        )
                        options.append(option_label)

                    selected = st.selectbox(
                        "Select a past conversation to load",
                        options,
                        index=0,
                        key="session_selector"
                    )

                    if st.button("📂 Load Selected Chat"):
                        selected_session_id = selected.split(" | ")[1]
                        selected_chat = chats.find_one(
                            {"session_id": selected_session_id})

                        if selected_chat:
                            st.session_state.session_id = selected_chat[
                                "session_id"]
                            st.session_state.chat_start_time = selected_chat[
                                "start_time"]
                            st.session_state.chat_history = selected_chat.get(
                                "messages", [])
                            logging.info(
                                f"✅ Loaded session: {selected_session_id}")
                            st.success(
                                f"✅ Loaded session: {selected_session_id}")
                            st.rerun()

                # st.session_state.req_type = "Login"

                if "session_id" not in st.session_state:
                    st.session_state.session_id = generate_chat_id()
                if "chat_start_time" not in st.session_state:
                    st.session_state.chat_start_time = datetime.now(
                        timezone.utc)
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                chats.insert_one({
                    "session_id": st.session_state.session_id,
                    "user_id": email,
                    "start_time": st.session_state.chat_start_time,
                    "messages": []
                })
                logging.info("✅ Logged in!")
                st.success("✅ Logged in!")
                st.rerun()
            else:
                logging.error("❌ Email not found. Please register.")
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
                st.session_state.chat_start_time = chat_data.get(
                    "start_time", datetime.now(timezone.utc))
                st.session_state.chat_history = chat_data.get("messages", [])
                logging.info("✅ Session restored!")
                st.success("✅ Session restored!")
                st.rerun()
            else:
                logging.error("❌ Invalid Ticket ID")
                st.error("❌ Invalid Ticket ID")

    elif login_mode == "🆕 Start New Chat":
        register_form()

    else:
        register_form()
else:
    user_details()

    # if "greeting_shown" not in st.session_state:
    #    greeting = find_default_answer("greeting")
    #    ## greeting == "I don't understand that.":
    #      #  greeting = "Hello! 👋 How can I assist you today?"
    #    st.info(greeting)
    #    st.session_state.greeting_shown = True

    chat_interface()
