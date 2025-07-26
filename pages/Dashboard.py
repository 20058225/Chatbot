import streamlit as st
import io
from services.mongo import db
from bson import ObjectId 
from collections import defaultdict
import altair as alt
import pandas as pd
from services.import_file import insert_data_streamlit
from datetime import date, datetime, timezone

print("🔧 Recarregando Dashboard.py...")


chats = db["chats"]
unanswered = db["unanswered"]
faq = db["faq"]
default_chat = db["default_chat"]
knowledge = db["knowledge"] 

st.set_page_config(page_title="Dashboard", page_icon="👩‍💻")
st.title("🧠 Chatbot Admin Dashboard")

with st.expander("➕ Add New Data"):
    insert_data_streamlit(faq, default_chat, knowledge)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 FAQs",  # quick questions
    "❓ Unanswered",  # display questions unanswered
    "📈 Chat Logs",  # Audit and export conversations
    "📄 Knowledge Articles", # manange articles 
    "📊 User Stats 💡"
])

def export_chat_history(chat_history):
    if not chat_history:
        st.info("No messages to export.")
        return
    df = pd.DataFrame(chat_history)
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Export Chat History as CSV",
        data=csv,
        file_name=f"chat_{st.session_state.session_id}.csv",
        mime="text/csv",
    )

with tab1:
    st.subheader("FAQs")
    data = list(faq.find())

    for doc in data:
        if "question" in doc and "answer" in doc:
            st.markdown(f"**Q:** {doc['question']}")
            st.write(f"**A:** {doc['answer']}")
        
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("📝 Edit", key=f"edit_{doc['_id']}"):
                    new_q = st.text_input("Edit Question", value=doc["question"], key=f"q_{doc['_id']}")
                    new_a = st.text_area("Edit Answer", value=doc["answer"], key=f"a_{doc['_id']}")
                    if st.button("✅ Save Changes", key=f"save_{doc['_id']}"):
                        faq.update_one({"_id": doc["_id"]}, {"$set": {"question": new_q, "answer": new_a}})
                        st.success("Updated!")
                        st.rerun()
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{doc['_id']}"):
                    faq.delete_one({"_id": doc["_id"]})
                    st.warning("Deleted!")
                    st.rerun()
            st.write("---")
        else: 
            st.warning(f"⚠️ Skipping malformed entry with ID: {doc.get('_id')}")

with tab2:
    st.subheader("Unanswered Questions")
    data = list(unanswered.find())
    for doc in data:
        st.markdown(f"**Question:** {doc['question']}")
        st.write(f"From: {doc['user_id']} | At: {doc['timestamp']}")

        if st.button("Add as FAQ", key=f"add_{doc['_id']}"):
        # Use a modal-like approach using session state to edit
            st.session_state['edit_faq'] = {
                "id": str(doc['_id']),
                "question": doc['question'],
                "answer": ""
            }
            st.rerun()
    
    if 'edit_faq' in st.session_state:
        faq = st.session_state['edit_faq']
        st.markdown(f"### Editing FAQ for: {faq['question']}")
        faq['answer'] = st.text_area("Answer", value=faq.get('answer', ''))
        if st.button("Save FAQ"):
            # Save FAQ to faq collection
            faq.insert_one({
                "question": faq['question'],
                "answer": faq['answer']
            })
            # Remove from unanswered
            unanswered.delete_one({"_id": ObjectId(faq['id'])})
            st.success("FAQ added successfully!")
            del st.session_state['edit_faq']
            st.rerun()

        if st.button("Cancel"):
            del st.session_state['edit_faq']
            st.rerun()
            
        st.write("---")

with tab3:
    st.subheader("📈 Chat Logs")

    # 🔍 Search bar for session data
    search_term = st.text_input("Search chat sessions", "")
    query = {}

    if search_term:
        query = {
            "$or": [
                {"user_id": {"$regex": search_term, "$options": "i"}},
                {"messages.question": {"$regex": search_term, "$options": "i"}},
                {"messages.answer": {"$regex": search_term, "$options": "i"}},
            ]
        }

    # Get list of session chats
    chat_sessions = list(chats.find(query).sort("last_updated", -1).limit(100))

    all_msgs = []  # Collect all logs across sessions

    if chat_sessions:
        for chat in chat_sessions:
            session_id = chat.get("session_id", "N/A")
            user_id = chat.get("user_id", "N/A")
            messages = chat.get("messages", [])
            start_time = chat.get("start_time")
            readable_time = start_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_time, datetime) else "unknown"

            st.markdown(f"### 👤 User: `{user_id}` | 🏷️ Session: `{session_id}` | 📅 Started: {readable_time}")

            export_rows = []  # For per-session export with metadata

            if messages:
                for i, msg in enumerate(messages):
                    question = msg.get("question", "N/A")
                    answer = msg.get("answer", "N/A")
                    sentiment = msg.get("sentiment", "N/A")
                    priority = msg.get("priority", "N/A")
                    timestamp = msg.get("timestamp")
                    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else str(timestamp)

                    st.markdown(f"**Q{i+1}:** {question}")
                    st.markdown(f"**A:** {answer}")
                    st.markdown(f"🧠 Sentiment: `{sentiment}`   🚨 Priority: `{priority}`   🕒 {ts_str}")
                    st.markdown("---")

                    row = {
                        "session_id": session_id,
                        "createdAt": readable_time,
                        "email": user_id,
                        "question": question,
                        "answer": answer,
                        "sentiment": sentiment,
                        "priority": priority,
                        "thumbs_up": msg.get("thumbs_up", False),
                        "thumbs_down": msg.get("thumbs_down", False),
                        "timestamp": ts_str
                    }

                    export_rows.append(row)
                    all_msgs.append(row)

                # 👇 Per-Session Export Button
                df_session = pd.DataFrame(export_rows)
                csv_session = df_session.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Export This Session as CSV",
                    data=csv_session,
                    file_name=f"{session_id}_chat.csv",
                    mime="text/csv",
                    key=f"export_{session_id}"
                )

            else:
                st.info("No messages found for this session.")

            st.markdown("----")

        # ✅ All-Sessions Export Button
        if all_msgs:
            df_all = pd.DataFrame(all_msgs)
            csv_all = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Export All Chat Logs as CSV",
                data=csv_all,
                file_name="all_chat_logs.csv",
                mime="text/csv"
            )
    else:
        st.info("No chat sessions found.")

with tab4:
    st.subheader("📄 Knowledge Articles")

    articles = list(knowledge.find())

    if articles:
        for article in articles:
            st.markdown(f"### 📘 {article.get('title', 'No Title')}")
            st.write(article.get("content", "No content provided"))
            st.markdown(f"🕒 Imported: {article.get('import_timestamp')}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✏️ Edit", key=f"edit_article_{article['_id']}"):
                    new_title = st.text_input("Edit Title", value=article.get("title", ""), key=f"title_{article['_id']}")
                    new_content = st.text_area("Edit Content", value=article.get("content", ""), key=f"content_{article['_id']}")
                    if st.button("✅ Save", key=f"save_article_{article['_id']}"):
                        knowledge.update_one(
                            {"_id": article["_id"]},
                            {"$set": {"title": new_title, "content": new_content}}
                        )
                        st.success("✅ Article updated!")
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_article_{article['_id']}"):
                    knowledge.delete_one({"_id": article["_id"]})
                    st.warning("🗑️ Article deleted!")
                    st.rerun()
            st.write("---")
    else:
        st.info("No knowledge articles found. Add one using the importer above.")

with tab5:
    st.subheader("📊 Total Sessions per User")
    stats = chats.aggregate([
        {"$group": {
            "_id": "$user_id", 
            "total_sessions": {
                "$sum": 1}
            }},
        {"$sort": {"total_sessions": -1}}
    ])

    data = list(stats)
    if data:
        df = pd.DataFrame(data).rename(columns={"_id": "User", "total_sessions": "Total Sessions"})
        st.table(df)
    else:
        st.info("No session data to show.")

    st.subheader("🎯 Message Priority Distribution")
    pipeline = [
        {"$unwind": "$messages"},
        {"$group": {
            "_id": "$messages.priority",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ]

    result = list(chats.aggregate(pipeline))
    if result:
        df = pd.DataFrame(result).rename(columns={"_id": "priority", "count": "count"})
        pie = alt.Chart(df).mark_arc().encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(field="priority", type="nominal"),
            tooltip=[alt.Tooltip("priority", title="Priority"), alt.Tooltip("count", title="Count")]
        ).properties(width=400, height=400)

        st.altair_chart(pie, use_container_width=True)
    else:
        st.info("No messages found for priority analysis.")
    
    st.subheader("🧠 Message Sentiment Distribution")
    sentiment_pipeline = [
        {"$unwind": "$messages"},
        {"$group": {"_id": "$messages.sentiment", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    sentiment_result = list(chats.aggregate(sentiment_pipeline))
    if sentiment_result:
        df_sentiment = pd.DataFrame(sentiment_result).rename(columns={"_id": "Sentiment", "count": "Count"})
        chart_sentiment = alt.Chart(df_sentiment).mark_arc().encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Sentiment", type="nominal"),
            tooltip=[alt.Tooltip("Sentiment"), alt.Tooltip("Count")]
        ).properties(width=400, height=400)
        st.altair_chart(chart_sentiment, use_container_width=True)
    else:
        st.info("No messages found for sentiment analysis.")

    st.subheader("👍 Thumbs Up / 👎 Thumbs Down Feedback Counts")
    
    # Aggregate thumbs up/down counts from chat messages
    feedback_pipeline = [
        {"$unwind": "$messages"},
        {"$group": {
            "_id": None,
            "thumbs_up": {"$sum": {"$cond": [{"$eq": ["$messages.thumbs_up", True]}, 1, 0]}},
            "thumbs_down": {"$sum": {"$cond": [{"$eq": ["$messages.thumbs_down", True]}, 1, 0]}}
        }}
    ]

    feedback_result = list(chats.aggregate(feedback_pipeline))
    if feedback_result:
        thumbs_up = feedback_result[0].get("thumbs_up", 0)
        thumbs_down = feedback_result[0].get("thumbs_down", 0)

        st.markdown(f"👍 Total Likes: `{thumbs_up}`        👎 Total Dislikes: `{thumbs_down}`")

        feedback_df = pd.DataFrame({
            "Feedback": ["Thumbs Up", "Thumbs Down"],
            "Count": [thumbs_up, thumbs_down]
        })
        
        chart_feedback = alt.Chart(feedback_df).mark_bar().encode(
            x=alt.X('Feedback', sort=None),
            y='Count',
            color='Feedback',
            tooltip=['Feedback', 'Count']
        ).properties(width=400, height=300)

        st.altair_chart(chart_feedback, use_container_width=True)
    else:
        st.info("No feedback data found.")