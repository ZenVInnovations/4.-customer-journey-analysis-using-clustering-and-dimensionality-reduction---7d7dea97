import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("CustomerbehaviourTourism.csv")

    # Convert columns and clean data (same as in your notebook)
    df['yearly_avg_Outstation_checkins'] = df['yearly_avg_Outstation_checkins'].replace('*', np.nan)
    df['yearly_avg_Outstation_checkins'] = pd.to_numeric(df['yearly_avg_Outstation_checkins'], errors='coerce')
    df['member_in_family'] = df['member_in_family'].replace('Three', 3)
    df['member_in_family'] = pd.to_numeric(df['member_in_family'], errors='coerce')

    df.fillna(df.median(numeric_only=True), inplace=True)

    return df

def preprocess_data(df, selected_columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[selected_columns])
    return scaled

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    return reduced, pca

def determine_optimal_clusters(data):
    inertia = []
    silhouette = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(data, labels))
    return K, inertia, silhouette

def cluster_and_plot(data, reduced_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='tab10')
    plt.title("Customer Segments via PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    st.pyplot(plt)
    return labels

# --- Streamlit UI ---

st.title("Customer Journey Analysis using Clustering and PCA")

df = load_data()
st.write("### Raw Data Preview")
st.write(df.head())

# Select numeric columns for clustering
numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop(columns=['UserID'], errors='ignore').columns.tolist()
selected_cols = st.multiselect("Select features for analysis", numeric_cols, default=numeric_cols[:5])

if selected_cols:
    scaled_data = preprocess_data(df, selected_cols)
    reduced_data, pca = apply_pca(scaled_data)

    st.write("### Explained Variance by PCA")
    st.bar_chart(pca.explained_variance_ratio_)

    st.write("### Optimal Number of Clusters")
    K, inertia, silhouette = determine_optimal_clusters(scaled_data)
    fig, ax1 = plt.subplots()
    ax1.plot(K, inertia, 'b-', label='Inertia')
    ax2 = ax1.twinx()
    ax2.plot(K, silhouette, 'r-', label='Silhouette')
    st.pyplot(fig)

    n_clusters = st.slider("Select number of clusters", 2, 10, value=4)
    labels = cluster_and_plot(scaled_data, reduced_data, n_clusters)
    df['Cluster'] = labels
    st.write("### Clustered Data Sample")
    st.write(df[['Cluster'] + selected_cols].head())
