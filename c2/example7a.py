import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Încărcăm setul de date Wine
# - load_wine() returnează un obiect cu .data (caracteristici) și .target (etichete)
wine = load_wine()
X = wine.data # matricea de caracteristici (13 coloane)
y = wine.target # vectorul de etichete (3 clase de vin)

# 2. Împărțim în seturi de antrenament și test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% din date merg în setul de test
    random_state=42 # seed fix pentru reproducibilitate
)

# 3. Definim grila de valori pentru hiperparametrul 'max_depth'
# - Vom testa adâncimi între 2 și 10 pentru arborele de decizie
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# 4. Configurăm GridSearchCV
# - estimator: DecisionTreeClassifier cu random_state pentru stabilitate
# - param_grid: lista de valori pentru 'max_depth'
# - cv=5: validare încrucișată cu 5 fold-uri
# - scoring='accuracy': criteriul de selecție este acuratețea medie pe fold-uri
# - n_jobs=-1: folosește toate nucleele CPU disponibile pentru paralelizare
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 5. Antrenăm GridSearchCV pe datele de antrenament
# - Pentru fiecare valoare din param_grid, se antrenează un arbore și se evaluează cu CV
grid_search.fit(X_train, y_train)

# 6. Afișăm cei mai buni parametri și scorul de validare încrucișată
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Val Score: {grid_search.best_score_:.3f}")
# - best_params_ conține valoarea lui 'max_depth' care a obținut cea mai mare acuratețe medie pe CV
# - best_score_ este acea acuratețe medie (pe cele 5 fold-uri)

# 7. Evaluăm modelul cu cei mai buni parametri pe setul de test
best_model = grid_search.best_estimator_ # arborul de decizie cu max_depth optim
y_pred = best_model.predict(X_test) # facem predicții pe X_test
test_accuracy = accuracy_score(y_test, y_pred) # calculăm acuratețea pe setul de test
print(f"Test Accuracy: {test_accuracy:.3f}")
# - Afișează acuratețea reală pe date pe care modelul nu le-a văzut la antrenament
