# pages/Dashboard.py
import streamlit as st
import pandas as pd
from bson import ObjectId
from pathlib import Path
from datetime import datetime, timezone
import altair as alt

from services.mongo import db
from services.ml import train_and_save_models_from_csv, train_and_save_kmeans_from_csv
from services.db import update_message_feedback 

st.set_page_config(page_title="📊 Chatbot Dashboard", page_icon="👩‍💻", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

st.title("📊 Chatbot Admin Dashboard")

# Mongo collections
chats = db["chats"]
monitoring = db["monitoring"]
unanswered = db["unanswered"]
faq = db["faq"]
default_chat = db["default_chat"]
knowledge = db["knowledge"]



tab1, tab2, tab3, tab4 = st.tabs([
    "📚 FAQs",
    "📄 Knowledge Articles",
    "💬 Chat History & Editor",
    "📊 User & Message Stats"
])

# === TAB 1: FAQs Management ===
with tab1:
    st.subheader("📚 FAQs")

    if st.button("➕ Add New FAQ"):
        st.session_state['adding_new_faq'] = True

    if st.session_state.get('adding_new_faq'):
        with st.form("add_new_faq_form"):
            new_q = st.text_input("Question")
            new_a = st.text_area("Answer")
            submitted = st.form_submit_button("Save New FAQ")
            cancelled = st.form_submit_button("Cancel")

            if submitted:
                faq.insert_one({"question": new_q, "answer": new_a})
                st.success("New FAQ added!")
                del st.session_state['adding_new_faq']
                st.rerun()

            if cancelled:
                del st.session_state['adding_new_faq']
                st.rerun()

    faqs = list(faq.find())
    for doc in faqs:
        st.markdown(f"**Q:** {doc.get('question', '')}")
        st.write(f"**A:** {doc.get('answer', '')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Edit", key=f"edit_faq_{doc['_id']}"):
                st.session_state['editing_faq_id'] = str(doc['_id'])
                st.session_state['editing_faq_question'] = doc.get('question', '')
                st.session_state['editing_faq_answer'] = doc.get('answer', '')
                st.rerun()
        with col2:
            if st.button("🗑️ Delete", key=f"delete_faq_{doc['_id']}"):
                faq.delete_one({"_id": doc["_id"]})
                st.success("FAQ deleted!")
                st.rerun()

        st.markdown("---")

    if st.session_state.get('editing_faq_id'):
        with st.form("edit_faq_form"):
            q = st.text_input("Edit question", value=st.session_state.get('editing_faq_question', ''))
            a = st.text_area("Edit answer", value=st.session_state.get('editing_faq_answer', ''))
            submitted = st.form_submit_button("Save FAQ")
            cancelled = st.form_submit_button("Cancel")

            if submitted:
                faq.update_one(
                    {"_id": ObjectId(st.session_state['editing_faq_id'])},
                    {"$set": {"question": q, "answer": a}}
                )
                st.success("FAQ updated!")
                del st.session_state['editing_faq_id']
                del st.session_state['editing_faq_question']
                del st.session_state['editing_faq_answer']
                st.rerun()
            if cancelled:
                del st.session_state['editing_faq_id']
                del st.session_state['editing_faq_question']
                del st.session_state['editing_faq_answer']
                st.rerun()

# === TAB 2: Knowledge Articles ===
with tab2:
    st.subheader("📄 Knowledge Articles")

     # Add New Article Button and Form
    if st.button("➕ Add New Article"):
        st.session_state['adding_new_article'] = True

    if st.session_state.get('adding_new_article'):
        with st.form("add_new_article_form"):
            new_t = st.text_input("Title")
            new_c = st.text_area("Content")
            submitted = st.form_submit_button("Save New Article")
            cancelled = st.form_submit_button("Cancel")

            if submitted:
                knowledge.insert_one({
                    "title": new_t,
                    "content": new_c,
                    "import_timestamp": datetime.now()
                })
                st.success("New article added!")
                del st.session_state['adding_new_article']
                st.rerun()
            if cancelled:
                del st.session_state['adding_new_article']
                st.rerun()

    articles = list(knowledge.find())
    for article in articles:
        st.markdown(f"### 📘 {article.get('title', 'No Title')}")
        st.write(article.get("content", "No content"))
        st.markdown(f"🕒 Imported: {article.get('import_timestamp', '')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✏️ Edit Article", key=f"edit_article_{article['_id']}"):
                st.session_state['editing_article_id'] = str(article['_id'])
                st.session_state['editing_article_title'] = article.get('title', '')
                st.session_state['editing_article_content'] = article.get('content', '')
                st.rerun()
        with col2:
            if st.button(f"🗑️ Delete Article", key=f"delete_article_{article['_id']}"):
                knowledge.delete_one({"_id": article["_id"]})
                st.success("Article deleted!")
                st.rerun()

        st.markdown("---")

    if st.session_state.get('editing_article_id'):
        with st.form("edit_article_form"):
            t = st.text_input("Title", value=st.session_state.get('editing_article_title', ''))
            c = st.text_area("Content", value=st.session_state.get('editing_article_content', ''))
            submitted = st.form_submit_button("Save Article")
            cancelled = st.form_submit_button("Cancel")

            if submitted:
                knowledge.update_one(
                    {"_id": ObjectId(st.session_state['editing_article_id'])},
                    {"$set": {"title": t, "content": c}}
                )
                st.success("Article updated!")
                del st.session_state['editing_article_id']
                del st.session_state['editing_article_title']
                del st.session_state['editing_article_content']
                st.rerun()
            if cancelled:
                del st.session_state['editing_article_id']
                del st.session_state['editing_article_title']
                del st.session_state['editing_article_content']
                st.rerun()

# === TAB 3: Chat History ===
with tab3:
    st.subheader("💬 Chat History")

    search_term = st.text_input("Search sessions by user ID or message content", key="search_chat")

    query = {}
    if search_term:
        query = {
            "$or": [
                {"user_id": {"$regex": search_term, "$options": "i"}},
                {"messages.question": {"$regex": search_term, "$options": "i"}},
                {"messages.answer": {"$regex": search_term, "$options": "i"}}
            ]
        }

    chat_sessions = list(chats.find(query).sort("start_time", -1).limit(50))

    if chat_sessions:
        for chat in chat_sessions:
            session_id = chat.get("session_id", "N/A")
            user_id = chat.get("user_id", "N/A")
            messages = chat.get("messages", [])
            start_time = chat.get("start_time")
            readable_time = start_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_time, datetime) else "unknown"

            with st.expander(f"👤 User: `{user_id}` | 🏷️ Session: `{session_id}` | 📅 Started: {readable_time}"):

                for msg in messages:
                    # Detecta sender e texto sem depender do campo sender
                    if msg.get("question"):
                        sender = "User"
                        text = msg["question"]
                    elif msg.get("answer"):
                        sender = "Bot"
                        text = msg["answer"]
                    else:
                        sender = "Unknown"
                        text = msg.get("text", "")

                    # Feedback baseado nos booleans atuais
                    if msg.get("thumbs_up"):
                        feedback = "positivo"
                    elif msg.get("thumbs_down"):
                        feedback = "negativo"
                    else:
                        feedback = ""

                    timestamp = msg.get("timestamp")
                    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else str(timestamp)

                    st.write(f"**{sender}:** {text}  _(at {ts_str})_")
                    # st.write(f"**{sender}:** {text}  _(at {ts_str})_  — Feedback: {feedback}")

    else:
        st.info("No chat sessions found.")

    st.markdown("---")
    st.info("🗄️ Quick MongoDB Chats Editor")

    df_chats = pd.DataFrame(list(chats.find()))
    if not df_chats.empty:
        df_chats["_id"] = df_chats["_id"].astype(str)
        edited_df = st.data_editor(df_chats, num_rows="dynamic")

        if st.button("💾 Save changes"):
            for _, row in edited_df.iterrows():
                doc_id = ObjectId(row["_id"])
                updated_doc = row.drop(labels=["_id"]).to_dict()
                chats.update_one({"_id": doc_id}, {"$set": updated_doc})
            st.success("Chats collection updated!")

    delete_id = st.text_input("Delete chat by _id (paste ObjectId here):")
    if st.button("Delete document"):
        try:
            oid = ObjectId(delete_id)
            result = chats.delete_one({"_id": oid})
            if result.deleted_count > 0:
                st.success("Document deleted!")
            else:
                st.warning("No document found with that _id.")
        except Exception as e:
            st.error(f"Invalid _id or error: {e}")


# === TAB 4: User & Message Stats ===
with tab4:        
    st.subheader("📊 User & Message Statistics")

    total_chats = chats.count_documents({})
    total_faqs = faq.count_documents({})
    total_knowledge = knowledge.count_documents({})

    st.write(f"Total chat sessions: {total_chats}")
    st.write(f"Total FAQs: {total_faqs}")
    st.write(f"Total Knowledge Articles: {total_knowledge}")

    st.markdown("---")

    # 🔹 Estatísticas de sentimento a partir de chats.messages
    pipeline_sentiment = [
        {"$unwind": "$messages"},
        {"$match": {"messages.sentiment": {"$exists": True}}},
        {"$group": {"_id": "$messages.sentiment", "count": {"$sum": 1}}}
    ]
    sentiment_counts = list(chats.aggregate(pipeline_sentiment))
    df_sentiment = pd.DataFrame(sentiment_counts).rename(columns={"_id": "Sentiment", "count": "Count"})

    if not df_sentiment.empty:
        chart = alt.Chart(df_sentiment).mark_bar().encode(
            x='Sentiment',
            y='Count',
            color='Sentiment'
        ).properties(title="Sentiment Distribution in Messages")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No sentiment data available.")

    # 🔹 Estatísticas de intent_tag e feedback a partir de chats.messages
    pipeline_intents = [
        {"$unwind": "$messages"},
        {"$match": {"messages.intent_tag": {"$exists": True}}},
        {
            "$group": {
                "_id": None,
                "avg_response_time": {"$avg": "$messages.bot_time"},
                "fallback_count": {"$sum": {"$cond": ["$messages.is_fallback", 1, 0]}},
                "thumbs_up_count": {"$sum": {"$cond": ["$messages.thumbs_up", 1, 0]}},
                "thumbs_down_count": {"$sum": {"$cond": ["$messages.thumbs_down", 1, 0]}},
                "unique_users": {"$addToSet": "$user_id"},
                "total_responses": {"$sum": 1}
            }
        }
    ]
    agg_result = list(chats.aggregate(pipeline_intents))

    if agg_result:
        data = agg_result[0]
        total_responses = data.get("total_responses", 1)
        fallback_rate = data.get("fallback_count", 0) / total_responses
        thumbs_up = data.get("thumbs_up_count", 0)
        thumbs_down = data.get("thumbs_down_count", 0)
        total_feedback = thumbs_up + thumbs_down
        satisfaction = ((thumbs_up - thumbs_down) / total_feedback) if total_feedback > 0 else 0
        unique_users = len(data.get("unique_users", []))
        avg_response_time = data.get("avg_response_time") or 0

        st.write(f"Total AI Responses Logged: {total_responses}")
        st.write(f"Unique Users Interacted: {unique_users}")
        st.write(f"Average AI Response Time: {avg_response_time:.3f} seconds")
        st.write(f"Fallback Rate: {fallback_rate:.2%}")
        st.write(f"Satisfaction Score (thumbs up minus down): {satisfaction:.2f}")

        df_feedback = pd.DataFrame({
            "Feedback": ["Thumbs Up", "Thumbs Down"],
            "Count": [thumbs_up, thumbs_down]
        })
        chart = alt.Chart(df_feedback).mark_bar().encode(
            x='Feedback',
            y='Count',
            color='Feedback'
        ).properties(title="Feedback Distribution")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No records with intent tags found in chat messages.")
