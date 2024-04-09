"""
    Script pour effectuer l'ingénierie des fonctionnalités sur les données.
"""
# Imports
from sklearn.model_selection import train_test_split

from create_signal import read_historical, read_all_stock, plot_close_time


# Rendement décalé
def lag_returns(df, n) -> list[str]:
    for i in range(1,n+1):
        df["return_lagged_by_"+str(i)] = df["return"].shift(i)
    return ["return_lagged_by_"+str(i) for i in range(1,n+1)]

# Préparation des données pour l'entraînement du modèle
def data_train_test(df, pr=False):
    """
    Calcul le rendement et ajoute 2 colonnes de rendements décalés
    Supprime les valeurs nulls
    Classifi les rendement négatif=0 et positif=1
    Sépare les données à 70% Train 30% Test
    Args:
        df (dataframe): données sources

    Returns:
        list: données Train et Test
    """
    print("--- data_train_test ---")
    
    df["return"] = df["close"].pct_change()
    if pr:
        print("Calcul du rendement:")
        print(df.head())
    
    data_lags = lag_returns(df, 2)
    if pr:
        print("Rendement décalé:")
        print(data_lags)
        print(df.head())
    
    df.dropna(inplace=True)
    if pr:
        print("dropna:")
        print(df.head())

    # Nombre de rendement positif et négatif
    df["variation"] = (df["return"] > 0).astype(int)
    if pr:
        print("Nombre de rendement positif et négatif:")
        print(df["variation"].value_counts())
    
    X = df[data_lags]
    y = df["variation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    return X_train, X_test, y_train, y_test, X, y



# ---------------- main --------------------------------
if __name__ == "__main__":
    df_hist = read_historical()    
    df_all_stock = read_all_stock()
    X_train, X_test, y_train, y_test, X, y = data_train_test(df_hist, True)
