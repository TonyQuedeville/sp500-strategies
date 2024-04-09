"""
    Script pour exécuter la recherche de grille sur les hyperparamètres du modèle.
"""

# Import
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from create_signal import read_historical, read_all_stock
from features_engineering import data_train_test

# Plot de la division en séries temporelles
def plot_timeSeries_split(split, n_splits=5):
    plt.figure(figsize=(10, 6))
    for i, (train_index, test_index) in enumerate(split, 1):
        plt.scatter(train_index, [i] * len(train_index), color='blue', label='Train' if i == 1 else None)
        plt.scatter(test_index, [i] * len(test_index), color='red', label='Test' if i == 1 else None)

    plt.xlabel('Sample Index')
    plt.ylabel('Split Iteration')
    plt.title('Time Series Split')
    plt.legend()
    plt.yticks(range(1, n_splits + 1), ['Split {}'.format(i) for i in range(1, n_splits + 1)])
    plt.show()

# Division des données en blocs
def block_split(X, n_splits=5, test_size=0.2):
    block_split = []
    block_size = int(len(X)/n_splits)
    indexes = [x*block_size for x in range(0,n_splits+1)]

    for i in range(len(indexes)-1):
        block = X.iloc[indexes[i]:indexes[i+1]]
        print(test_size * len(block))
        split_index = int(test_size * len(block))
        block_split.append([block[split_index:], block[:split_index]]) 
        
    return block_split

def plot_block_split(block_split):
    plt.figure(figsize=(10, 6))
    for i, (train, test) in enumerate(block_split, 1):
        train_index = train.index
        test_index = test.index
        plt.scatter(train_index, [i] * len(train_index), color='blue', label='Train' if i == 1 else None)
        plt.scatter(test_index, [i] * len(test_index), color='red', label='Test' if i == 1 else None)

    plt.xlabel('Sample Index')
    plt.ylabel('Split Iteration')
    plt.title('Indexes in Custom Block Split')
    plt.legend()
    plt.yticks(range(1, len(block_split) + 1), ['Split {}'.format(i) for i in range(1, len(block_split) + 1)])
    plt.show()


# Lire le fichier CSV
def read_cv_results(file_path):
    cv_results_df = pd.read_csv(file_path)
    return cv_results_df

# Recherche sur la grille des hyperparamètres
def grid_search(X, y, pr=False):
    """
    Cette fonction effectue une recherche sur la grille des hyperparamètres pour trouver les meilleurs paramètres pour un modèle 
    donné (dans ce cas, un RandomForestClassifier). Elle utilise la validation croisée pour évaluer les performances du modèle avec 
    différents ensembles de paramètres et enregistre les résultats dans un fichier CSV.

    Args:
        X (Dataframe): 
        y (serie): 
        pr (bool, optional): affichage des résultats intermediaires. Defaults to False.
    """
    model = RandomForestClassifier()

    # grid search parameters
    forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(1,14))}]
    metrics = {"accuracy": "accuracy", "AUC": "roc_auc"}

    gridsearch = GridSearchCV(
        model,
        forest_params,
        scoring=metrics,
        refit="AUC",
        n_jobs=2,
        cv=10,
        error_score='raise',
        return_train_score=True,
    )
    gridsearch.fit(X,y)

    if pr:
        print('best estimator : ', gridsearch.best_estimator_)
        print('best params : ', gridsearch.best_params_)
        print('best score : ', gridsearch.best_score_)
        # print('cross-validation results : ', gridsearch.cv_results_)
        
    # Enregistrer les résultats dans un fichier CSV
    cv_results_df = pd.DataFrame(gridsearch.cv_results_)
    cv_results_df.to_csv('../results/cross-validation.csv', index=False)


def split(X, y, pr=False):
    # time series split
    n_splits = 5
    time_series_split = TimeSeriesSplit(n_splits=n_splits)
    plot_timeSeries_split(time_series_split.split(X), n_splits)
    
    # block series split
    data = X
    data["prediction"] = y
    split_blocks = block_split(X, n_splits=5, test_size=0.3)

    if pr:
        print("TRAIN : " ,len(split_blocks[0][0]))
        print("TEST : ", len(split_blocks[0][1]))
    
    plot_block_split(split_blocks)
    
    X.drop("prediction", axis=1 ,inplace=True)
    
    if pr:
        print(X.head())
        print(y.value_counts())
        print("X:", X.shape)
    
    # Index
    folds = {}
    indexes = []
    for i, (train, test) in enumerate(split_blocks):
        indexes.append(("fold_"+str(i), "train"))
        indexes.append(("fold_"+str(i), "test"))

        folds["train_"+str(i)] = i # replace i by the auc
        folds["test_"+str(i)] = i

    indexes_pd = pd.MultiIndex.from_tuples(indexes, names=["set", "AUC"])
    if pr:
        print("indexes_pd:", indexes_pd)

    df_indexes = pd.DataFrame(folds, index=indexes)
    if pr:
        print("df_indexes:")
        print(df_indexes.head(10))  


# ---------------- main --------------------------------
if __name__ == "__main__":
    df_hist = read_historical()    
    df_all_stock = read_all_stock()
    X_train, X_test, y_train, y_test, X, y = data_train_test(df_hist)
    split(X, y, True)
    grid_search(X, y, True)