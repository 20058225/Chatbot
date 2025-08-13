# pages/Monitoring.py
import streamlit as st
import altair as alt
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import logging
import os
import csv
import plotly.express as px
from simulation.simulate_chat_tests import run_tests
from pathlib import Path
from services.mongo import db
from services.monitoring import load_logs
from services.ml import train_and_save_models_from_csv, train_and_save_kmeans_from_csv

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chatbot-monitoring.log") # sempre o mesmo nome

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="Monitoring & Tests", page_icon="📊", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

st.title("📈 Monitoring & Automated Tests")

# MongoDB collections
monitoring_col = db["monitoring"]
test_results_col = db["test_results"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Train Models",
    "🔍 Monitoring Logs",
    "🧪 Test Results",
    "Execution Metrics", 
    "Monitoring Logs"
])

# --- TAB 1: Train Models ---
with tab1:
    st.subheader("📤 Upload CSV & Train Models")

    uploaded_file = st.file_uploader("Upload a CSV file to train the models", type=["csv"])

    if uploaded_file:
        csv_path = DATA_DIR / uploaded_file.name
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        success_box1 = st.success(f"✅ CSV saved at `{csv_path}`")
        time.sleep(5)
        success_box1.empty()

        df = pd.read_csv(csv_path)
        st.dataframe(df.head())

        total = len(df)
        info_box1 = st.info(f"📊 The file contains **{total} lines** that will be processed.")


        if st.button("🚀 Train Models"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(total):
                time.sleep(0.5)  # Simula processamento
                pct = int(((i + 1) / total) * 100)
                progress_bar.progress(pct)
                status_text.text(f"Processing step {i+1}/{total} ({pct}%)")
            # ✅ Finalizou
            progress_bar.empty()   # 🔹 Remove a barra
            status_text.empty()    # 🔹 Remove o texto
            info_box1.empty()

            success_box2 = st.success("✅ Processing completed!")
            time.sleep(3)
            success_box2.empty()

            with st.spinner("Training models..."):
                try:
                    train_and_save_models_from_csv(csv_path)
                    train_and_save_kmeans_from_csv(csv_path)
                    monitoring_col.insert_one({
                        "event": "train_models",
                        "file": str(csv_path),
                        "timestamp": datetime.now(timezone.utc),
                        "status": "success",
                        "log_source": "production"
                    })
                    success_box3 = st.success("✅ Models trained and saved in ml/models/")
                    time.sleep(3)
                    success_box3.empty()
                except Exception as e:
                    monitoring_col.insert_one({
                        "event": "train_models",
                        "file": str(csv_path),
                        "timestamp": datetime.now(timezone.utc),
                        "status": "error",
                        "error": str(e),
                        "log_source": "production"
                    })
                    st.error(f"❌ Error training models: {e}")

# --- TAB 2: Monitoring Logs ---
with tab2:  
    st.markdown("### 🔍 Monitoring Logs — Filter & Export")

    collections = {
        "Monitoring (prod)": "monitoring",
        "Test Results (simulation)": "test_results"
    }
    selected_collection_label = st.selectbox(
        "📂 Select the Collection",
        options=list(collections.keys()),
        index=0
    )
    selected_collection = db[collections[selected_collection_label]]

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        user_filter = st.text_input("Filter by user_id (regex)")
    with col2:
        model_filter = st.text_input("Filter by model (exact)")
    with col3:
        date_from = st.date_input("From", value=None)
        date_to = st.date_input("To", value=None)
    with col4:
        log_source_filter = st.selectbox(
            "Source", options=["All", "production", "simulation"], index=0)

    query = {}
    if user_filter:
        query["user_id"] = {"$regex": user_filter, "$options": "i"}
    if model_filter:
        query["model"] = model_filter
    if date_from:
        query.setdefault("timestamp", {})["$gte"] = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    if date_to:
        query.setdefault("timestamp", {})["$lte"] = datetime.combine(date_to, datetime.max.time(), tzinfo=timezone.utc)
    if log_source_filter != "All":
        query["log_source"] = log_source_filter

    logs = list(selected_collection.find(query).sort("timestamp", -1).limit(5000))
    st.write(f"Showing {len(logs)} log rows (limit 5000) from **{selected_collection_label}**.")

    if logs:
        df_logs = pd.DataFrame(logs)
        if "_id" in df_logs.columns:
            df_logs["_id"] = df_logs["_id"].astype(str)
        if "timestamp" in df_logs.columns:
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        if "details" in df_logs.columns:
            df_logs["details"] = df_logs["details"].astype(str)

        st.dataframe(df_logs.head(400))

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export filtered monitoring logs as CSV", 
            data=csv, 
            file_name="monitoring_logs.csv", 
            mime="text/csv")
    else:
        st.info("No logs match the filter.")

# --- TAB 3: Test Results ---
with tab3:
    st.subheader("🧪 Chatbot Test Results")

    uploaded_questions = st.file_uploader("📤 Upload question file (.txt) to run the test", type=["txt"])

    if uploaded_questions and st.button("🚀 Run Tests"):

        questions_path = Path("data") / uploaded_questions.name
        with open(questions_path, "wb") as f:
            f.write(uploaded_questions.getvalue())

        with open(questions_path, "r", encoding="utf-8") as f:
            questions_list = [line.strip() for line in f if line.strip()]

        total = len(questions_list)
        info_box2 = st.info(f"📊 The file contains **{total} lines** that will be processed.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, q in enumerate(questions_list):
            # Aqui vai sua lógica de teste por pergunta
            time.sleep(0.5)  # Simula tempo de processamento
            pct = int((i + 1) / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"Processing step {i+1}/{total} ({pct}%)")

        progress_bar.empty()   # 🔹 Remove a barra
        status_text.empty()    # 🔹 Remove o texto
        info_box2.empty()

        success_box4 = st.success("✅ Processing completed!")
        time.sleep(3)
        success_box4.empty()

        results = run_tests(str(questions_path))
        success_box5 = st.success("✅ Tests performed!")
        time.sleep(3)
        success_box5.empty()

        df_up = pd.DataFrame(results)
        if "_id" in df_up.columns:
            df_up["_id"] = df_up["_id"].astype(str)
        if "timestamp" in df_up.columns:
            df_up["timestamp"] = pd.to_datetime(df_up["timestamp"])
        st.dataframe(df_up)

    # Model filter (pega valores reais distintos presentes na collection)
    model_options = ["All"] + sorted(
        test_results_col.distinct("model", {"log_source": "simulation"})
    )
    model_filter = st.selectbox("Filter by model", options=model_options, index=0)
    query = {"log_source": "simulation"}
    if model_filter != "All":
        query["model"] = model_filter

    results = list(test_results_col.find(query).sort("timestamp", -1))

    if results:
        df = pd.DataFrame(results)
        if "_id" in df.columns:
            df["_id"] = df["_id"].astype(str)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df)

    st.markdown("---")

# --- Resumo de métricas ---
    st.markdown("### 📊 Summary Metrics")

    if "execution_time" in df.columns and "model" in df.columns:
        avg_gpt2_time = df.loc[df["model"] == "gpt-2", "execution_time"].mean()
        avg_bert_time = df.loc[df["model"] == "bert", "execution_time"].mean()
        st.write(f"Average GPT-2 Response Time: {avg_gpt2_time:.3f} seconds")
        st.write(f"Average BERT Response Time: {avg_bert_time:.3f} seconds")

    # Se existir coluna de "correct" ou similar, calcule acurácia média (exemplo)
    if "correct" in df.columns and df["correct"].dtype in [bool, int, float]:
        accuracy = df["correct"].mean()
        st.write(f"Overall Accuracy: {accuracy:.2%}")

    # --- Gráfico 1: Boxplot dos tempos por modelo ---
    chart_response_time = alt.Chart(
        df[df["model"] == "gpt-2"]
    ).mark_boxplot().encode(
        x=alt.X("model:N", title="Model"),
        y=alt.Y("execution_time:Q", title="GPT-2 Response Time (s)"),
        color="model:N"
    ).properties(
        title="GPT-2 Response Time Distribution"
    )
    st.altair_chart(chart_response_time, use_container_width=True)

    chart_bert_time = alt.Chart(
        df[df["model"] == "bert"]
    ).mark_boxplot().encode(
        x=alt.X("model:N", title="Model"),
        y=alt.Y("execution_time:Q", title="BERT Response Time (s)"),
        color="model:N"
    ).properties(
        title="BERT Response Time Distribution"
    )
    st.altair_chart(chart_bert_time, use_container_width=True)

    # --- Gráfico 2: Acurácia média por modelo (se disponível) ---
    if "correct" in df.columns and df["correct"].dtype in [bool, int, float]:
        df_acc = df.groupby("model")["correct"].mean().reset_index()
        chart_accuracy = alt.Chart(df_acc).mark_bar().encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("correct:Q", title="Accuracy"),
            color="model:N"
        ).properties(title="Accuracy by Model")
        st.altair_chart(chart_accuracy, use_container_width=True)

    # Botão para apagar resultados
    confirm_delete = st.checkbox("Are you sure you want to delete ALL test results?")
    if confirm_delete and st.button("🗑️ Confirm Delete All Test Results"):
        res = test_results_col.delete_many(query)
        success_box6 = st.success(f"Deleted {res.deleted_count} test result documents.")
        time.sleep(2)
        success_box6.empty()
        st.rerun()
    # else:
      # st.info("No test results found in the database.")


##### -----
with tab4:
    st.subheader("Execution Metrics Overview")
    log_type = st.selectbox("Select log type:", ["all", "train", "test"])
    df = load_logs(log_type)

    if not df.empty:
        if "execution_time" in df.columns and "model" in df.columns:
            avg_gpt2_time = df.loc[df["model"] == "gpt-2", "execution_time"].mean()
            avg_bert_time = df.loc[df["model"] == "bert", "execution_time"].mean()

            st.write(f"Average GPT-2 Response Time: {avg_gpt2_time:.3f} seconds")
            st.write(f"Average BERT Response Time: {avg_bert_time:.3f} seconds")
    else:
        st.warning("No logs found for the selected type.")

    st.markdown("---")

    st.subheader("Comparative Charts")
    log_type = st.selectbox("Select log type for charts:", ["all"])
    
    df_all = load_logs(log_type)
    if df_all.empty:
        st.warning("No data for charts.")
    else:
        # 🔹 Conversão de ObjectId para string para evitar erro no Streamlit/PyArrow
        import bson
        # 🔹 Garantir conversão de timestamp para datetime (se existir)
        if "timestamp" in df_all.columns:
            try:
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
            except Exception:
                pass

        # === MÉDIA DE TEMPO POR MODELO ===
        if "execution_time" in df_all.columns and df_all["execution_time"].notna().any():
            avg_time = df_all.groupby("model")["execution_time"].mean().reset_index()
            fig_time = px.bar(
                avg_time,
                x="model",
                y="execution_time",
                title="Average Execution Time per Model"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("⚠️ No 'execution_time' data available for chart.")

        # === MÉDIA DE SCORE POR MODELO ===
        if "score" in df_all.columns and df_all["score"].notna().any():
            avg_score = df_all.groupby("model")["score"].mean().reset_index()
            fig_score = px.bar(
                avg_score,
                x="model",
                y="score",
                title="Average Score per Model"
            )
            st.plotly_chart(fig_score, use_container_width=True)
        else:
            st.info("⚠️ No 'score' data available for chart.")

        # === EVOLUÇÃO TEMPORAL DO TEMPO DE EXECUÇÃO ===
        if "execution_time" in df_all.columns and "timestamp" in df_all.columns:
            fig_line = px.line(
                df_all.dropna(subset=["execution_time"]),
                x="timestamp",
                y="execution_time",
                color="model",
                title="Execution Time Over Runs"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("⚠️ Missing columns for execution time over time chart.")


with tab5:
    st.subheader("Monitoring Logs")
    log_type = st.selectbox("Select log type for details:", ["all","train", "test"], key="logs_tab2")
    df_logs = load_logs(log_type)

    if not df_logs.empty:
        st.dataframe(df_logs)
    else:
        st.warning("No detailed logs found.")