# services/import_file.py

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
import json

def insert_data_streamlit(request_col, default_chat_col, knowledge_col=None):
    st.subheader("📥 Import Data into Database")

    import_method = st.radio(
        "📂 How would you like to import data?",
        ["Upload CSV or JSON File", "Manual Entry"],
        key="import_method_radio"
    )

    data_type = st.selectbox(
        "🗂️ Select data type to import",
        ["FAQs / Tutorials", "Default Messages", "Knowledge Articles"],
        key="data_type_select"
    )

    if import_method == "Upload CSV or JSON File":
        uploaded_file = st.file_uploader(
            "Upload your CSV or JSON file", type=["csv", "json"], key="file_uploader"
        )

        if uploaded_file:
            file_name = uploaded_file.name

            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)

                if detect_format(df, data_type):
                    st.success(f"✅ Correct format detected for {data_type}")
                else:
                    st.warning("⚠️ Could not validate columns for this data type.")

                if st.button("📥 Import Data", key="import_csv"):
                    import_from_dataframe(df, data_type, request_col, default_chat_col, knowledge_col)

            elif file_name.endswith(".json"):
                try:
                    data = json.load(uploaded_file)
                    if isinstance(data, dict):
                        data = [data]
                    st.json(data)

                    if st.button("📥 Import Data", key="import_json"):
                        import_from_list(data, data_type, request_col, default_chat_col, knowledge_col)
                except Exception as e:
                    st.error(f"❌ Failed to parse JSON: {e}")

    elif import_method == "Manual Entry":
        if data_type == "FAQs / Tutorials":
            with st.form("faq_manual_form"):
                question = st.text_input("Question")
                answer = st.text_area("Answer")
                if st.form_submit_button("➕ Add"):
                    if question and answer:
                        record = base_record()
                        record["question"] = question
                        record["answer"] = answer
                        request_col.insert_one(record)
                        st.success("✅ FAQ added!")

        elif data_type == "Default Messages":
            with st.form("default_msg_form"):
                msg_type = st.text_input("Type")
                message = st.text_area("Message")
                if st.form_submit_button("➕ Add"):
                    if msg_type and message:
                        record = base_record()
                        record["type"] = msg_type.lower()
                        record["message"] = message
                        default_chat_col.insert_one(record)
                        st.success("✅ Default message added!")

        elif data_type == "Knowledge Articles":
            with st.form("manual_entry_knowledge"):
                title = st.text_input("Title")
                content = st.text_area("Article Content")
                if st.form_submit_button("➕ Add"):
                    if title and content and knowledge_col:
                        record = base_record()
                        record["title"] = title
                        record["content"] = content
                        knowledge_col.insert_one(record)
                        st.success("✅ Knowledge article added!")


def detect_format(df, data_type):
    if data_type == "FAQs / Tutorials":
        return "question" in df.columns and "answer" in df.columns
    elif data_type == "Default Messages":
        return "type" in df.columns and "message" in df.columns
    elif data_type == "Knowledge Articles":
        return "title" in df.columns and "content" in df.columns
    return False


def base_record():
    return {
        "import_timestamp": datetime.now(timezone.utc),
        "import_source": "manual"
    }


def import_from_dataframe(df, data_type, request_col, default_chat_col, knowledge_col):
    data = df.to_dict(orient="records")
    for item in data:
        item.update(base_record())

    if data_type == "FAQs / Tutorials":
        request_col.insert_many(data)
    elif data_type == "Default Messages":
        default_chat_col.insert_many(data)
    elif data_type == "Knowledge Articles" and knowledge_col is not None:
        knowledge_col.insert_many(data)
    st.success(f"✅ Imported {len(data)} items to {data_type}!")


def import_from_list(data, data_type, request_col, default_chat_col, knowledge_col):
    for item in data:
        item.update(base_record())

    if data_type == "FAQs / Tutorials":
        request_col.insert_many(data)
    elif data_type == "Default Messages":
        default_chat_col.insert_many(data)
    elif data_type == "Knowledge Articles" and knowledge_col is not None:
        knowledge_col.insert_many(data)
    st.success(f"✅ Imported {len(data)} items to {data_type}!")