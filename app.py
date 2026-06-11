import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.title("Customer Segmentation Dashboard")

uploaded_file = st.file_uploader(
    "Upload Customer Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    st.subheader("Cluster Summary")
    st.dataframe(df.groupby("Cluster").mean(numeric_only=True))

    score = silhouette_score(X_scaled, clusters)

    st.subheader("Silhouette Score")
    st.write(score)

    fig, ax = plt.subplots()

    ax.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster']
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segmentation")

    st.subheader("Customer Segments")
    st.pyplot(fig)

    st.subheader("Business Insights")

    st.write("""
    • High Income + High Spending → Premium Customers

    • High Income + Low Spending → Potential Customers

    • Low Income + High Spending → Frequent Shoppers

    • Low Income + Low Spending → Budget Customers
    """)