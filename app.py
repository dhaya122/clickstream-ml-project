import streamlit as st
import pandas as pd
import joblib

st.title("Clickstream Conversion Prediction App (Optimized)")

model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
le = joblib.load("label_encoder.pkl")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview")
    st.dataframe(df.head())

    if "page" in df.columns:
        X = df.drop(columns=["page"])

        X_p = preprocessor.transform(X)

        preds = model.predict(X_p)
        preds_labels = le.inverse_transform(preds)

        st.subheader("Predictions")
        st.write(preds_labels)

        st.subheader("Distribution")
        st.bar_chart(pd.Series(preds_labels).value_counts())

    else:
        st.error("Target column 'page' not found")