import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.pipeline import make_pipeline
from src.data import get_all_concepts

st.title('Anomalous U.S. Public Companies')

headers = {
    'User-Agent': 'Andrew Abeles andrewabeles@sandiego.edu'
}

@st.cache
def load_data(headers, period, schema):
    data = get_all_concepts(headers, period, schema)
    return data

@st.cache
def load_pipeline():
    pipeline = make_pipeline(
        PowerTransformer(method='yeo-johnson', standardize=True),
        KNNImputer(weights='distance'),
        PCA(n_components=6, random_state=1)
    )
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
df_final = pd.concat([df_raw.reset_index(), df_processed], axis=1).copy()
df_final['anomaly_strength'] = model.reachability_
df_final = df_final.loc[model.ordering_]
df_final['cluster_ordering'] = np.arange(0, len(df_final))

st.metric('Number of Reporting Companies', len(df_raw))

eps = st.select_slider(
    'Select Anomaly Threshold',
    options=np.arange(0, 30.1, 0.1),
    value=3.5
)
clusters = extract_clusters(model, eps=eps)
df_final['cluster'] = clusters

col1, col2 = st.columns(2)
with col1: 
    st.subheader('Reachability Plot')
    fig = px.scatter(
        data_frame=df_final,
        x='cluster_ordering',
        y='anomaly_strength',
        color='cluster',
        labels={
            'anomaly_strength': 'Anomaly Strength',
            'cluster': 'Cluster'
        },
        hover_data={
            'entityName': True,
            'cluster_ordering': False,
            'anomaly_strength': False,
            'cluster': False
        }
    )
    fig.add_hline(y=eps)
    st.plotly_chart(fig)
with col2:
    st.subheader('Cluster Distribution')
    cluster_dist = pd.DataFrame(df_final['cluster'].value_counts().reset_index())
    fig = px.bar(
        data_frame=cluster_dist,
        y='index',
        x='cluster',
        color='index',
        orientation='h',
        labels={
            'index': '',
            'cluster': 'Companies'
        },
        hover_data={
            'cluster': True,
            'index': False
        }
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

st.subheader('Scatterplot of Clustering')
col1, col2 = st.columns(2)
with col1:
    x = st.selectbox(
        'Select X-Axis',
        options=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'],
        index=0
    )
with col2:
    y = st.selectbox(
        'Select Y-Axis',
        options=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'],
        index=1
    )
fig = px.scatter(
    data_frame=df_final,
    x=x,
    y=y,
    color='cluster',
    opacity=0.3,
    hover_data={
        'entityName': True,
        'cik': True,
        'cluster': False,
        'PC1': False,
        'PC2': False,
        'PC3': False,
        'PC4': False,
        'PC5': False,
        'PC6': False
    }
)
st.plotly_chart(fig)

st.subheader('Principal Components')
#pipeline['pca'].

st.subheader('Anomalous Companies')
anomalies = df_final.query("cluster == 'anomalous'").sort_values('anomaly_strength', ascending=False)
st.write(anomalies)
