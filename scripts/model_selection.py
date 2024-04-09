"""
    Script pour sélectionner le meilleur modèle à partir des résultats de la recherche de grille.
"""

# Import
import imp
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from create_signal import read_historical, read_all_stock, plot_close_time
from features_engineering import data_train_test


# Entraînement et évaluation du modèle de régression logistique
def log_reg(df, X_train, X_test, y_train, pr=False):
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    
    X_test["prediction"] = model.predict(X_test)
    X_test["return"] = df["return"][X_test.index[0]:]
    
    if pr:
        print(X_test["prediction"].value_counts())
    
        X_test["gain"] = X_test["prediction"] * X_test["return"]
        (X_test[["gain", "return"]] + 1).cumprod().plot()
        plt.show()
    
    return model


# ---------------- main --------------------
if __name__ == "__main__":
    df_hist = read_historical()    
    df_all_stock = read_all_stock()
    X_train, X_test, y_train, y_test, X, y = data_train_test(df_hist)
    log_reg(df_hist, X_train, X_test, y_train, True)