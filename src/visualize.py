import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import cluster_optics_dbscan

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

def univ_dist(x, **kwargs):
    """Given an array, this function prints its descriptive statistics and plots its distribution."""
    print(x.describe())
    fig, (ax1, ax2) = plt.subplots(ncols=2, **kwargs)
    sns.kdeplot(x=x, ax=ax1)
    sns.violinplot(x=x, ax=ax2)
    plt.show()

def biv_dist(X, xname, yname, **kwargs):
    """Given two column names, this function prints their correlation coefficient and displays them in a scatterplot."""
    r = round(X.corr().loc[xname, yname], 3)
    print('Correlation Coefficient: {0}'.format(r))
    fig, ax = plt.subplots(**kwargs)
    sns.scatterplot(
        data=X, 
        x=xname,
        y=yname,
        ax=ax
    )
    plt.show()

def plot_corr_matrix(X, **kwargs):
    """Given a dataframe, this function calculates its correlation matrix and displays it as a triangular heatmap."""
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # mask for the upper triangle 
    fig, ax = plt.subplots(**kwargs)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1, # maximum possible correlation coefficient 
        vmin=-1, # minimum possible correlation coefficient 
        center=0, # center of possible correlation coefficients 
        annot=True, # display correlation coefficients 
        fmt=".2f", # round correlation coefficients to 2 decimal places
        ax=ax
    )
    return fig

def plot_loadings(pca_results, component, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    ax = pca_results['loadings'].loc[component].sort_values().plot(kind='barh')
    plt.xlabel('Loading')
    title = "{0} Loadings".format(component)
    plt.title(title)
    plt.show()

def reachability_plot(optics, eps=np.inf, clusters=None, **kwargs):
    x = np.arange(len(optics.labels_))
    reachability = optics.reachability_
    ordering = optics.ordering_
    fig, ax = plt.subplots(**kwargs)
    sns.scatterplot(
        x=x,
        y=reachability[ordering],
        hue=clusters[ordering],
        alpha=0.3,
        linewidth=0,
        ax=ax
    )
    ax.plot(np.full_like(x, eps, dtype=float), "k-", alpha=0.5, label='threshold')
    plt.legend()
    ax.set_ylabel('Reachability (epsilon distance)')
    return fig
