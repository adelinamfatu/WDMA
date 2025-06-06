import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Generăm un set de date sintetic cu două caracteristici numerice
np.random.seed(42)
num_samples = 50
X = np.random.rand(num_samples, 2) * 10 # Matricea X (50, 2), valori între 0 și 10

# Definim relația "adevărată" pentru țintă:
# y = 3 * Feature1 + 2 * Feature2 + zgomot
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)
# - Adăugăm zgomot normal cu sigma=5 pentru realism

# Convertim într-un DataFrame pentru claritate
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y
#   - df conține 3 coloane: 'Feature1', 'Feature2' și 'Target'

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[["Feature1", "Feature2"]] # DataFrame cu cele două caracteristici
y = df["Target"] # Serie cu valorile țintă

# 3. Împărțim datele în seturi de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# - test_size=0.3: 30% dintre exemple pentru evaluare
# - random_state=42: împărțire fixă pentru reproducibilitate

# 4. Definim modelul DecisionTreeRegressor și grila de hiperparametri pentru tuning
tree = DecisionTreeRegressor(random_state=42)
param_grid = {
    "max_depth": [2, 4, 6], # valori posibile pentru adâncimea maximă a arborelui
    "min_samples_leaf": [1, 2, 4] # numărul minim de exemple dintr-o frunză
}

# 5. Configurăm GridSearchCV
grid_search = GridSearchCV(
    estimator=tree, # modelul de bază (arbore de decizie)
    param_grid=param_grid, # grila de hiperparametri definită mai sus
    cv=3, # validare încrucișată pe 3 fold-uri
    scoring="r2", # folosim R² pentru a compara modele
    n_jobs=-1 # folosește toate nucleele CPU disponibile
)

# 6. Antrenăm căutarea prin grilă pe setul de antrenament
grid_search.fit(X_train, y_train)
# - pentru fiecare combinație din param_grid, se face validare încrucișată pe 3 fold-uri

# 7. Aflăm cei mai buni hiperparametri și evaluăm modelul optim pe setul de test
print("Best Parameters:", grid_search.best_params_)
# - best_params_ este un dicționar cu valorile hiperparametrilor care au dat cel mai bun scor R²

best_model = grid_search.best_estimator_
# - best_estimator_ este instanța modelului antrenată cu hiperparametrii optimi pe întregul set de antrenament

# Realizăm predicții și calculăm metricile pe setul de test
y_pred_test = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred_test) 
mse = mean_squared_error(y_test, y_pred_test) 
mae = mean_absolute_error(y_test, y_pred_test)

print(f"Test Set R²: {r2:.3f}")
print(f"Test Set MSE: {mse:.3f}")
print(f"Test Set MAE: {mae:.3f}")