import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.pipeline import make_pipeline
from src.data import get_all_concepts

st.set_page_config(layout="wide")

st.title('Anomalous U.S. Public Companies')
about_expander = st.expander('Click here to learn about this app')
with about_expander:
    """
    This app detects anomalous U.S. public companies based on their core financials disclosed to the SEC for a given year. 
    It uses the following financial metrics: 
    - Assets (USD)
    - Current Assets (USD)
    - Cash and Cash Equivalents at Carrying Value (USD)
    - Common Stock Authorized Shares (number of shares)
    - Common Stock Issued Shares (number of shares)
    - Common Stock Value (USD)
    - Liabilities and Stockholder Equity (USD)
    - Liabilities (USD)
    - Current Liabilities (USD)
    - Retained Earnings or Accumulated Deficit (USD)
    - Stockholders Equity (USD)  

    It retrieves data for the given year from the SEC's Extensible Business Markup Language APIs. 
    These APIs are rate-limited, so please be patient if your request for a given year's data is temporarily blocked.

    Once retrieved, the app transforms the raw data and estimates its missing values. It then clusters the transformed data 
    and labels companies which do not belong to a cluster as 'anomalous.' The user-chosen threshold determines how many 
    companies to consider anomalous. The anomalous companies' data is displayed in the table below and can be downloaded as a CSV.
    """

headers = {
    'User-Agent': 'Andrew Abeles andrewabeles@sandiego.edu'
}

FEATURE_NAMES = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']

@st.cache
def load_data(headers, period, schema):
    data = get_all_concepts(headers, period, schema)
    return data

@st.cache(allow_output_mutation=True)
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
        columns=FEATURE_NAMES
    )
    return df_processed, pipeline['pca'].components_

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

@st.cache
def df_to_csv(df):
    return df.to_csv().encode('utf-8')

def plot_loadings(loadings, component):
    fig = px.bar(
        data_frame=loadings.sort_values(component),
        x=component,
        y='index',
        orientation='h',
        labels={
            'index': '',
            component: 'Correlation'
        },
        hover_data={
            'index': False
        },
        title=component
    )
    return fig

def year_to_period(year):
    return 'CY' + str(year) + 'Q4I'

year = st.select_slider(
    'Select Year',
    options=np.arange(2012, 2022),
    value=2020,
    help='Select which year of disclosed financials to retrieve from the SEC.'
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
df_processed, pca_components = process_data(pipeline, df_raw)
loadings = pd.DataFrame(
    pca_components.T,
    columns=FEATURE_NAMES,
    index=pipeline.feature_names_in_
).reset_index() 
model = fit_model(df_processed)
df_final = pd.concat([df_raw.reset_index(), df_processed], axis=1).copy()
df_final['anomaly_strength'] = model.reachability_
df_final = df_final.loc[model.ordering_]
df_final['cluster_ordering'] = np.arange(0, len(df_final))

eps = st.select_slider(
    'Select Anomaly Threshold',
    options=np.arange(0, 30.1, 0.1),
    value=3.5,
    help='Select a threshold reachability value above which companies will be classified as outliers. Reachability represents how different a company is from those most similar.'
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
            'cluster_ordering': 'Cluster Ordering',
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

st.subheader('Cluster Visualization')
col1, col2 = st.columns(2)
with col1:
    x = st.selectbox(
        'Select X-Axis',
        options=FEATURE_NAMES,
        index=0
    )

with col2:
    y = st.selectbox(
        'Select Y-Axis',
        options=FEATURE_NAMES,
        index=1,
        help="""Select which principal components to visualize. Principal components are higher-level representations of the original variables.
                The bar plots below show the components' correlations with the original variables and are used to understand what each component represents."""
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

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_loadings(loadings, x))

with col2:
    st.plotly_chart(plot_loadings(loadings, y))

st.subheader('Anomalous Companies')
anomalies = df_final.query("cluster == 'anomalous'").sort_values('anomaly_strength', ascending=False)
anomalies_csv = df_to_csv(anomalies)

col1, col2 = st.columns(2)
with col1:
    anomaly_name = st.selectbox(
        'Select an Anomalous Company',
        options=anomalies['entityName'].unique(),
        help='Select an anomalous company to view its principal component values. You can also view its raw financials by searching with CTRL-F in the table below.'
    )

anomaly_data = anomalies.query("entityName == @anomaly_name")
anomaly_data_pc = anomaly_data[FEATURE_NAMES].T
anomaly_data_pc.columns = ['value']

with col2:
    fig = px.bar(
        data_frame=anomaly_data_pc,
        x='value',
        y=anomaly_data_pc.index,
        orientation='h',
        labels={
            'index': ''
        },
        title=anomaly_name
    )
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig)

st.write('All Anomalous Companies')
st.write(anomalies.style.format(precision=2))

st.download_button(
    'Download CSV',
    data=anomalies_csv,
    file_name='anomalous_companies.csv',
    mime='text/csv'
)