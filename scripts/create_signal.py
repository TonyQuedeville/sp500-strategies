"""
    Script pour générer le signal d'apprentissage automatique à partir des données d'entrée.
"""

# Imports
from pickle import TRUE
import pandas as pd
import matplotlib.pyplot as plt

# Lecture des données historique
def read_historical(pr=False):
    print("--- read historical ---")
    df_hist = pd.read_csv("../data/HistoricalPrices.csv")
    df_hist['Date'] = pd.to_datetime(df_hist['Date'], format='%m/%d/%y')
    df_hist.set_index('Date', inplace=True)
    df_hist.sort_index(inplace=True, ascending=True)
    df_hist.columns = df_hist.columns.str.lower()
    
    if pr:
        print(df_hist.head())
        
        print("isnull ?")
        print(df_hist.isnull().sum())
    
    return df_hist

# Lecture de toutes les actions
def read_all_stock(pr=False):
    print("--- read all stock ---")
    df_all_stock = pd.read_csv('../data/all_stocks_5yr.csv')
    df_all_stock.head()
    df_all_stock['date'] = pd.to_datetime(df_all_stock['date'])
    df_all_stock.set_index('date', inplace=True)
    df_all_stock.sort_index(inplace=True)
    df_all_stock.columns = df_all_stock.columns.str.lower()
    
    if pr:
        print(df_all_stock.head())
        print("isnull ?")
        print(df_all_stock.isnull().sum())
    
    return df_all_stock

# Plot des données de clôture au fil du temps
def plot_close_time(df1, df2, title):
    fig, ax = plt.subplots()
    ax.plot(df1["close"], label='Historical')
    ax.plot(df2["close"], label='All stock')
    ax.set(ylabel='close', title=title)
    ax.legend() 
    ax.grid()

    plt.show()

# ---------------- main --------------------------------
if __name__ == "__main__":
    df_hist = read_historical(True)    
    df_all_stock = read_all_stock(True)
    plot_close_time(df_hist, df_all_stock, "Close/Time Comparaison")