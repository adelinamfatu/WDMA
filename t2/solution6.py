import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Încărcăm setul de date Wine
wine = load_wine()
X = wine.data # matricea de caracteristici (fiecare rând e un eșantion de vin cu 13 atribute)
y = wine.target # vectorul țintă cu valorile 0, 1 sau 2 pentru cele 3 clase de vin

# 2. Împărțim datele în seturi de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% din date merg în setul de test
    random_state=42 # seed pentru reproducibilitate
)

# 3. Definim grila de valori pentru hiperparametrii Logistic Regression
# - 'C' este inversul intensității regularizării (C mic → regularizare puternică)
# - 'penalty' alege între L1 (lasso) și L2 (ridge)
# - 'solver': 'saga' suportă atât L1 cât și L2 (și convergența pe seturi mai mari)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'] # folosim 'saga' pentru a suporta atât L1, cât și L2
}

# 4. Configurăm GridSearchCV cu validare încrucișată pe 5 fold-uri
# - estimator: un LogisticRegression cu max_iter=2000 pentru a asigura convergența
# - param_grid: grila de hiperparametri definită mai sus
# - cv=5: 5 fold-uri de validare încrucișată
# - n_jobs=-1: folosește toate nucleele CPU disponibile pentru a accelera căutarea
# - scoring='accuracy': criteriul de selecție este acuratețea medie pe fold-uri
grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# 5. Antrenăm GridSearchCV pe datele de antrenament
# - Pentru fiecare combinație din param_grid, se face validare încrucișată
grid_search.fit(X_train, y_train)

# 6. Afișăm cei mai buni hiperparametri găsiți și scorul mediu aferent
print("Best Parameters found by grid search:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.3f}")

# 7. Luăm modelul antrenat cu cei mai buni parametri și îl evaluăm pe setul de test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Model: {test_accuracy:.3f}")
