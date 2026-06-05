import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Clickstream Dashboard", layout="wide")

st.title("Clickstream Conversion Dashboard")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

uploaded_file = st.file_uploader("Upload Clickstream CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    if "page" not in df.columns:
        st.error("Column 'page' not found in dataset")
        st.stop()

    X = df.drop(columns=["page"])

    
    X_p = preprocessor.transform(X)
    preds = model.predict(X_p)
    preds_labels = le.inverse_transform(preds)

    df["Prediction"] = preds_labels

    
    st.subheader("Key Performance Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", len(df))
    col2.metric("Conversions", sum(df["Prediction"] == "Converted"))
    col3.metric("Conversion Rate (%)",
                round((df["Prediction"] == "Converted").mean() * 100, 2))

    
    st.subheader("Insights")

    chart_data = df["Prediction"].value_counts()

    st.bar_chart(chart_data)

    st.subheader("Pie View")
    st.pyplot(chart_data.plot.pie(autopct="%1.1f%%").figure)

    
    st.subheader("Download Results")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="clickstream_predictions.csv",
        mime="text/csv"
    )