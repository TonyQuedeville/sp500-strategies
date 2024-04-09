"""
    Script pour exécuter le backtesting de la stratégie financière basée sur le signal d'apprentissage automatique.
"""

# Import
import imp
import pickle
from gridsearch import read_cv_results

from create_signal import read_historical, read_all_stock
from features_engineering import data_train_test
from model_selection import log_reg

# Meilleur estimation
def bet_estimator(model, pr=False):
    grid = read_cv_results('../results/cross-validation.csv')
    if pr:
        print("grid:")
        print(grid.keys())
        print("nb mean_train_AUC:", len(grid["mean_train_AUC"]))
    
    with open("../results/best_estimator.pkl", "wb") as file:
        pickle.dump(model, file)


# ---------------- main --------------------
if __name__ == "__main__":
    df_hist = read_historical()    
    df_all_stock = read_all_stock()
    X_train, X_test, y_train, y_test, X, y = data_train_test(df_hist)
    model = log_reg(df_hist, X_train, X_test, y_train)
    bet_estimator(model)