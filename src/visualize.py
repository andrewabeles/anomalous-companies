import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_missing_data(df, **kwargs):
    """Plots a dataframe's proportion of missing values by column as a horizontal bar chart."""
    missing_data_summary = pd.DataFrame(
        df.isna().sum() / len(df),
        columns=['missing']
    ).sort_values('missing', ascending=False)
    fig, ax = plt.subplots(**kwargs)
    sns.barplot(
        data=missing_data_summary,
        y=missing_data_summary.index,
        x='missing',
        color='red',
        orient='h',
        ax=ax
    )
    plt.title('Missing Data by Column')
    plt.xlabel('Proportion Missing')
    plt.show()
