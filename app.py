import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from src.data import get_all_concepts
from src.visualize import plot_corr_matrix

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

def year_to_period(year):
    return 'CY' + str(year) + 'Q4I'

year = st.select_slider(
    'Select Year',
    options=np.arange(2000, 2022),
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

fig, ax = plt.subplots()
df_processed.plot(x='PC1', y='PC2', kind='scatter', ax=ax)
st.pyplot(fig)