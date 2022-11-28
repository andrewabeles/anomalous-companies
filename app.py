import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from src.data import get_all_concepts
from src.visualize import plot_corr_matrix, reachability_plot

st.title('Anomalous U.S. Public Companies')

headers = {
    'User-Agent': 'Andrew Abeles andrewabeles@sandiego.edu'
}

@st.cache
def load_data(headers, period, schema):
    data = get_all_concepts(headers, period, schema)
    return data

@st.cache(allow_output_mutation=True) 
def load_pipeline():
    with open('models/pipeline.p', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache
def process_data(pipeline, df_raw):
    X_processed = pipeline.fit_transform(df_raw)
    df_processed = pd.DataFrame(
        X_processed,
        columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']
    )
    return df_processed

@st.cache
def fit_model(df_processed, eps=3.5):
    model = OPTICS(min_samples=0.01, eps=eps, cluster_method='dbscan')
    model.fit(df_processed)
    return model

@st.cache
def extract_clusters(model, eps):
    clusters = cluster_optics_dbscan(
        reachability=model.reachability_,
        core_distances=model.core_distances_,
        ordering=model.ordering_,
        eps=eps
    )
    clusters = pd.Series(clusters).astype(str).apply(lambda x: 'anomalous' if x == '-1' else 'cluster' + x)
    return clusters

def year_to_period(year):
    return 'CY' + str(year) + 'Q4I'

year = st.select_slider(
    'Select Year',
    options=np.arange(2012, 2022),
    value=2020
)

period = year_to_period(year)

schema = pd.DataFrame({
    'taxonomy': np.repeat('us-gaap', 11),
    'tag': [
        'AssetsCurrent',
        'Assets',
        'CashAndCashEquivalentsAtCarryingValue',
        'CommonStockSharesAuthorized',
        'CommonStockSharesIssued',
        'CommonStockValue',
        'LiabilitiesAndStockholdersEquity',
        'LiabilitiesCurrent',
        'Liabilities',
        'RetainedEarningsAccumulatedDeficit',
        'StockholdersEquity'
    ],
    'unit': [
        'USD',
        'USD',
        'USD',
        'shares',
        'shares',
        'USD',
        'USD',
        'USD',
        'USD',
        'USD',
        'USD'
    ]
})

df_raw = load_data(headers, period, schema)
pipeline = load_pipeline()
df_processed = process_data(pipeline, df_raw)
model = fit_model(df_processed)
df_processed['anomaly_strength'] = model.reachability_

st.metric('Number of Reporting Companies', len(df_raw))

eps = st.select_slider(
    'Select Anomaly Threshold',
    options=np.arange(0, 30.1, 0.1),
    value=3.5
)
clusters = extract_clusters(model, eps=eps)
df_processed['cluster'] = clusters
df_final = pd.concat([df_raw.reset_index(), df_processed], axis=1)

col1, col2 = st.columns(2)
with col1: 
    st.subheader('Reachability Plot')
    fig = reachability_plot(model, eps=eps, clusters=clusters)
    st.pyplot(fig)
with col2:
    st.subheader('Cluster Distribution')
    fig, ax = plt.subplots()
    df_processed['cluster'].value_counts().sort_values().plot(kind='barh', ax=ax)
    for i in ax.containers:
        ax.bar_label(i,)
    ax.set_xlabel('Companies')
    st.pyplot(fig)

st.subheader('Scatterplot of First Two Principal Components')
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_processed,
    x='PC1',
    y='PC2',
    hue='cluster',
    alpha=0.3,
    ax=ax
)
st.pyplot(fig)

st.subheader('Anomalous Companies')
anomalies = df_final.query("cluster == 'anomalous'").sort_values('anomaly_strength', ascending=False)
st.write(anomalies)