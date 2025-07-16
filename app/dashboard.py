import streamlit as st
import io
from services.mongo import db
from collections import defaultdict
import pandas as pd
from services.import_file import insert_data_streamlit
from datetime import date, datetime, timezone

chats = db["chats"]
unanswered = db["unanswered"]
requests = db["requests"]
default_chat = db["default_chat"]


st.set_page_config(page_title="Dashboard", page_icon="👩‍💻")
st.title("🧠 Chatbot Admin Dashboard")

with st.expander("➕ Add New Data"):
    insert_data_streamlit(requests, default_chat)

tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "❓ Unanswered", "📈 Chat Logs"])

with tab1:
    st.subheader("FAQs / Tutorials")
    data = list(requests.find())

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
                        requests.update_one({"_id": doc["_id"]}, {"$set": {"question": new_q, "answer": new_a}})
                        st.success("Updated!")
                        st.rerun()
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{doc['_id']}"):
                    requests.delete_one({"_id": doc["_id"]})
                    st.warning("Deleted!")
                    st.rerun()
            st.write("---")
        else: 
            st.warning(f"⚠️ Skipping malformed entry with ID: {doc.get('_id')}")

with tab2:
    st.subheader("Unanswered Questions")
    data = list(unanswered.find())
    for doc in data:
        st.markdown(f"**Q:** {doc['question']}")
        st.write(f"From: {doc['user_id']} | At: {doc['timestamp']}")
        st.write("---")

with tab3:
    st.subheader("Chat Logs")

    seach_term = st.text_input("🔍 Search conversations", "")
    query = {}

    if seach_term:
        query = {
            "$or": [
                {"question": {"$regex": seach_term, "$options": "i"}},
                {"answer": {"$regex": seach_term, "$options": "i"}},
                {"user_id": {"$regex": seach_term, "$options": "i"}}
            ]
        }
    
    results = list(chats.find(query).sort("timestamp", -1).limit(100))

    filtered_results = [doc for doc in results if "timestamp" in doc and isinstance(doc["timestamp"], datetime)]

    if filtered_results:
        df_export = pd.DataFrame(filtered_results)
        df_export["timestamp"] = pd.to_datetime(df_export["timestamp"])
        
        if df_export["timestamp"].dt.tz is None:
            df_export["timestamp"] = df_export["timestamp"].dt.tz_localize("UTC")
        df_export["timestamp"] = df_export["timestamp"].dt.tz_convert(None)

        st.download_button("⬇️ Download as CSV", data=df_export.to_csv(index=False), file_name="chat_logs.csv", mime="text/csv")

        grouped = defaultdict(list)
        for doc in filtered_results:
            ts = doc.get("timestamp")
            if isinstance(ts, datetime):
                date_str = ts.strftime("%Y-%m-%d")
            else:
                date_str = str(ts).split("T")[0]
            grouped[(doc["user_id"], date_str)].append(doc)

        for (user_id, date), entries in grouped.items():
            st.markdown(f"###👤 {user_id} | 📅 {date}")
            for doc in entries:
                st.markdown(f"**User:** {doc['user_id']}")
                st.markdown(f"**Q**: {doc['question']}")
                st.markdown(f"**A**: {doc['answer']}")
                st.write(f"🕒 {doc['timestamp']}")
            st.write("---")
    else:
        st.info("No chat logs found with valid timestamps.")