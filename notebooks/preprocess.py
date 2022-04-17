import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

def load_data() -> pd.DataFrame:
    return pd.read_csv('../Data/winequality-red.csv')

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def normalize_data(df:pd.DataFrame, target_col_name:str) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df.drop(target_col_name, axis=1))

def plot_data(df:pd.DataFrame, target_col_name:str):
    plt.figure(figsize=(16,16))

    col_names = [col for col in df.columns if col != target_col_name]

    for i, col in enumerate(col_names):
        plt.subplot(4, 3, i+1)
        sns.barplot(x=target_col_name, y=col, data=df)
        plt.xlabel(target_col_name)
        plt.ylabel(col)
        plt.title(f'{col} vs {target_col_name}')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
    plt.show()
