import streamlit as st
import pandas as pd
from datetime import datetime, timezone

def insert_data_streamlit(request_col, default_chat_col):
    st.subheader("📥 Import Data into Database")

    import_method = st.radio("How would you like to import data?", ["Upload CSV File", "Manual Entry"], key="import_method_radio")
    data_type = st.selectbox("Select data type", ["FAQs / Tutorials", "Default Messages"], key="data_type_select")

    if import_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your CSV or JSON file", type=["csv", "json"], key="file_uploader")

        if uploaded_file is not None:
            file_name = uploaded_file.name

            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)

                if "question" in df.columns and "answer" in df.columns and data_type == "FAQs / Tutorials":
                    st.success("Detected FAQ format (question/answer columns)")
                elif "type" in df.columns and "message" in df.columns and data_type == "Default Messages":
                    st.success("Detected default_chat format (type/message columns)")
                else:
                    st.warning("Unrecognized format for selected data type.")

                if st.button("📥 Import Data", key="import_csv"):
                    data = df.to_dict(orient="records")
                    for record in data:
                        record["import_timestamp"] = datetime.now(timezone.utc)
                        record["import_source"] = "file"
                    if data_type == "FAQs / Tutorials":
                        request_col.insert_many(data)
                    else:
                        default_chat_col.insert_many(data)
                    st.success("✅ Data imported successfully!")
        
            elif file_name.endswith(".json"):
                import json
                file_content = uploaded_file.read()
                try:
                    data = json.loads(file_content)

                    if isinstance(data, dict):
                        data = [data]

                    st.json(data)

                    if st.button("📥 Import Data", key="import_json"):
                        for record in data:
                            record["import_timestamp"] = datetime.now(timezone.utc)
                            record["import_source"] = "file"
                        if data_type == "FAQs / Tutorials":
                            request_col.insert_many(data)
                        else:
                            default_chat_col.insert_many(data)
                        st.success("✅ JSON imported successfully!")

                except Exception as e:
                    st.error(f"❌ Failed to parse JSON: {e}")
                
    elif import_method == "Manual Entry":
        if data_type == "FAQs / Tutorials":
            with st.form("manual_entry_form_faq"):
                question = st.text_input("Question")
                answer = st.text_area("Answer")
                submitted = st.form_submit_button("Add to Database")

                if submitted and question and answer:
                    record = {
                        "question": question,
                        "answer": answer,
                        "import_timestamp": datetime.now(timezone.utc),
                        "import_source": "manual"
                    }
                    request_col.insert_one(record)
                    st.success("✅ Manual entry added!")
        else:
            with st.form("manual_entry_form_default"):
                msg_type = st.text_input("Type")
                message = st.text_area("Message")
                submitted = st.form_submit_button("Add to Database")

                if submitted and msg_type and message:
                    record = {
                        "type": msg_type.lower(),
                        "message": message,
                        "import_timestamp": datetime.now(timezone.utc),
                        "import_source": "manual"
                    }
                    default_chat_col.insert_one(record)
                    st.success("✅ Default message added!")
