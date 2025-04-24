import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Title
st.title("Customer Revenue Prediction")

# Upload data
uploaded_file = st.file_uploader("Upload customer features CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'customer_id' in df.columns:
        X = df.drop(columns=['customer_id'])
    else:
        X = df

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load or train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, np.random.uniform(500, 5000, size=len(X_scaled)))  # Dummy training (replace with real model)

    # Predict
    predictions = model.predict(X_scaled)

    # Show results
    df['Predicted Revenue'] = predictions
    st.subheader("Predicted Revenue")
    st.dataframe(df)

    # Option to download results
    csv = df.to_csv(index=False)
    st.download_button("Download Predictions", csv, "predicted_revenue.csv", "text/csv")
else:
    st.info("Please upload a CSV file to get started.")