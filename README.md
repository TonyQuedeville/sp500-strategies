# sp500-strategies
Projet IA: Zone01


Dans ce projet, nous appliquerons la machine au financement. L'objectif est de créer une stratégie financière basée sur un signal émis par un modèle d'apprentissage automatique qui surpasse le SP500.

Le projet est divisé en plusieurs parties :
1 - create_signal.py : Lecture des données sources. (téléchargez dans le dossier "data" HistoricalData.csv et all_stocks_5yr.csv)
2 - features_engineering.py : Préparation des données pour l'entrainement
3 - model_selection.py : Entrainement du modèle.
4 - gridsearch.py : Evalue les performances du modèle avec différents paramètres et enregistre les résultats dans cross-validation.csv
5 - strategy.py : Enregistre le meilleur modèle dans un fichier best_estimator.pkl réutilisable pour de futurs prédictions.
