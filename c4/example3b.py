import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs

# 1. Generăm date sintetice
# - make_blobs creează 500 de puncte (n_samples) distribuite în jurul a 4 centre în 2D, cu seed pentru reproducibilitate
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
# - 'X' este un array (500, 2) cu coordonatele punctelor
# - '_' sunt etichetele generate de make_blobs, pe care nu le folosim în clustering nesupravegheat

# 2. Definim un scorer personalizat pentru Silhouette în DBSCAN
# NOTE: Semnătura funcției trebuie să fie (estimator, X, y=None) pentru GridSearchCV
def dbscan_silhouette_scorer(estimator, X, y=None):
    # După ce GridSearchCV a apelat estimator.fit(X_train), etichetele de cluster sunt în estimator.labels_
    labels = estimator.labels_ # DBSCAN stochează etichetele atribuite fiecărui punct aici

    # Excludem outlierii (cei cu label == -1)
    mask = labels != -1
    # Dacă rămân mai puțin de două clustere (după excluderea outlierilor), Silhouette nu se poate calcula
    if len(set(labels[mask])) < 2:
        return -1 # Returnăm un scor invalid fix (−1) pentru a penaliza configurația

    # Calculăm Silhouette Score doar pentru punctele ne-outlieri
    return silhouette_score(X[mask], labels[mask])

# 3. Definim grila de hiperparametri pentru DBSCAN
param_grid = {
    'eps': [0.2, 0.3, 0.4, 0.5], # raza de vecinătate
    'min_samples': [3, 5, 7] # numărul minim de puncte pentru a forma un cluster dens
}

# 4. Construim un CV „single-split” care folosește întregul set ca antrenament și test
# În clustering nesupravegheat, adesea nu avem set de test separat
indices = np.arange(len(X))
single_split_cv = [(indices, indices)]
# - Tuplu (train_idx, test_idx) unde ambele conțin toate indexurile din X

# 5. Configurăm GridSearchCV
grid_search = GridSearchCV(
    estimator=DBSCAN(), # modelul DBSCAN
    param_grid=param_grid, # grila de parametri definită anterior
    scoring=dbscan_silhouette_scorer, # scor personalizat pentru Silhouette
    cv=single_split_cv, # folosim „single-split” CV
    n_jobs=-1 # folosește toate nucleele CPU disponibile
)

# 6. Rulăm căutarea prin grilă pe întregul set de date
grid_search.fit(X)
# - Pentru fiecare combinație de (eps, min_samples), DBSCAN.fit(X) e apelat
# - Apoi Silhouette Score este calculat prin scorer-ul definit

# 7. Afișăm cei mai buni hiperparametri și scorul asociat
print("Best Params:", grid_search.best_params_)
print("Best Silhouette Score:", grid_search.best_score_)