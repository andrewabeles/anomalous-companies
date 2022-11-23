import streamlit as st
import pandas as pd
import numpy as np
from src.data import get_all_concepts

headers = {
    'User-Agent': 'Andrew Abeles andrewabeles@sandiego.edu'
}

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

df = get_all_concepts(headers, period, schema)

df