import streamlit as st
import yagmail
import random
import os
import openai
from services.mongo import db
from datetime import datetime, timezone
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

email_admin = os.getenv("EMAIL_ADMIN")
email_pass = os.getenv("EMAIL_PASS")
users = db["users"]
chats = db["chats"]
requests = db["requests"]
unanswered = db["unanswered"]
default_chat = db["default_chat"]

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🧠 Intelligent Support Chatbot")

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    

def get_openai_reply(user_input):
    try:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                { "role": "system", "content": "You are a helpful assistant."},
                { "role": "user", "content": user_input}
            ]
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, there was an error with the AI: {e}"
    
user_input = st.text_input("Ask me anything")
if st.button("Ask AI"):
    if user_input:
        st.write(get_openai_reply(user_input))

def get_default_response(response_type):
    doc = default_chat.find_one({"intents": {"%exists": True}})
    if not doc:
        return "I don't understand that."
    
    for intent in doc["intents"]:
        if intent["tag"].lower() == response_type.lower():
            return random.choice(intent.get("responses", ["I don't understand that."]))
        return "I don't understand that."

# def find_known_answer(user_input):
 #   print(f"User input: {user_input}")
  #  doc = requests.find_one({"question": {"$regex": user_input, "$options": "i"}})
   # print(f"Matched doc:{doc}")
    #return doc["answer"] if doc else None

def send_email(to, subject, body):
    try:
        yag = yagmail.SMTP(email_admin, email_pass)
        yag.send(to=to, subject=subject, contents=body)
    except Exception as e:
        print("Email failed:", e)
        st.warning("⚠️ Failed to send email notification.")

def handle_unanswered(user, question, request_type="unknown"):
    unanswered.insert_one({
        "user_id": user["email"],
        "question": question,
        "request_type": request_type,
        "timestamp": datetime.now(timezone.utc)
    })
    # Notify admin
    subject = f"[Chatbot - Unanswered] New question from {user['email']}"
    body = f"Question: {question}\nType: {request_type}\nUser: {user['name']} ({user['email']})"
    send_email(email_admin, subject, body)

def handle_deslike(user, question, answer):
    subject = f"[Chatbot - Feedback] 👎 from {user['email']}"
    body = f"User: {user['name']} ({user['email']})\nDisliked Answer\n\nQ: {question}\nA: {answer}"
    send_email(email_admin, subject, body)

def log_conversation(user, question, answer):
    session_id = st.session_state.session_id
    timestamp = datetime.now(timezone.utc)

    chats.update_one(
        {"session_id": session_id},
        {
            "$push": {
                "messages": {
                    "question": question,
                    "answer": answer,
                    "timestamp": timestamp
                    }
            },
            "$set": {
                "last_updated": timestamp
            }
        },
        upsert = True
    )

def register_form():
    with st.form("register_form"):
        st.subheader("🔐 Start Chat")
        email = st.text_input("Your Email")
        name = st.text_input("Your Name")
        req_type = st.selectbox("What do you need help with?", ["Tutorial", "FAQ", "Other"])
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

def find_intent_response(user_input):
    user_input = user_input.strip().lower()
    doc = default_chat.find_one({"intents": {"$exists": True}})
    if not doc or "intents" not in doc:
        return None
    
    for intent in doc["intents"]:
        for pattern in intent.get("patterns", []):
            if pattern.lower() in user_input.lower():
                return random.choice(intent.get("responses", []))
    return None

def chat_interface():
    st.markdown("---")
    st.subheader("🤖 Chat with the Bot")

    if st.session_state.get("input_processed", False):
        st.session_state.input_processed = False
        st.session_state.user_input = ""

    for i, msg in enumerate(st.session_state.chat_history):
        q_time = msg.get("time")
        if q_time:
            time_str = q_time.strftime("%H:%M:%S")
        else:
            time_str = "now"
        st.markdown(f"👩‍💻 **You**: ({time_str}): {msg['q']}")
        st.markdown(f"🤖 **Bot**: ({time_str}): {msg['a']}")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("👍", key=f"like_{i}"):
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("👎", key=f"dislike_{i}"):
                st.success("Thanks, we'll improve that!")
                handle_deslike(st.session_state.user, msg["q"], msg["a"])
    
    user_input = st.text_input("Type your question and press Enter", key="user_input")
    if user_input and not st.session_state.get("input_processed", False):
        print(f"User input: {user_input}")

        answer = find_known_answer(user_input)
        if answer:
            print(f"Matched FAQ: {answer}")
        else:
            answer = find_intent_response(user_input)
            if answer:
                print(f"Matched Intent: {answer}")
            else:
                print("No match found in FAQ or Intent")
                answer = get_default_response("fallback")
                handle_unanswered(st.session_state.user, user_input, st.session_state.req_type)
        
        st.session_state.chat_history.append({"q": user_input, "a": answer, "time": datetime.now(timezone.utc)})
        log_conversation(st.session_state.user, user_input, answer)

        st.session_state.input_processed = True
        st.rerun()

if not st.session_state.user:
    st.subheader("🔐 Start or Resume a Chat")

    has_account = st.checkbox("Already have an account?")

    if has_account:
        email = st.text_input("Email")
        if st.button("Login"):
            user = users.find_one({"email": email})
            if user:
                users.update_one({"email": email}, {"$set": {"last_active": datetime.now(timezone.utc)}})
                st.session_state.user = user
                st.session_state.req_type = "Login"
                st.success("✅ Logged in!")

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
                
                st.rerun()        
            else:
                st.warning("❌ Email not found. Please register.")
    else:
        register_form()
else:
    user_details()

    if len(st.session_state.chat_history) == 0:
        greeting = get_default_response("greeting")
        if greeting == "I don't understand that.":
            greeting = "Hello! 👋 How can I assist you today?"
        st.info(greeting)

    chat_interface()