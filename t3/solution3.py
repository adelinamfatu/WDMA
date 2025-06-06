import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Generăm un set de date sintetic cu trei caracteristici numerice
np.random.seed(42) # pentru reproducibilitatea zgomotului
num_samples = 30
X = np.random.rand(num_samples, 3) * 10 # matrice (30, 3) cu valori între 0 și 10

# Definim relația "adevărată" pentru țintă:
# Target = 2*Feature1 + 0.5*(Feature2)^2 - 3*Feature3 + zgomot
true_y = 2 * X[:, 0] + 0.5 * (X[:, 1] ** 2) - 3 * X[:, 2]
noise = np.random.normal(0, 5, size=num_samples) # zgomot normal (media=0, sigma=5)
y = true_y + noise # ținta cu zgomot adăugat

# 2. Convertim într-un DataFrame pentru claritate și separăm X și y
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df["Target"] = y
# - df conține 4 coloane: Feature1, Feature2, Feature3 și Target

X = df[["Feature1", "Feature2", "Feature3"]] # DataFrame cu cele trei caracteristici
y = df["Target"] # Serie cu valorile țintă

# 3. Împărțim datele în set de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 4. Creăm și antrenăm arborele de decizie pentru regresie
# - max_depth=4 limitează adâncimea arborelui, prevenind un oarecare overfitting
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=4)
tree_reg.fit(X_train, y_train)
# - .fit() construiește arborele alegând split-uri care reduc cât mai mult MSE

# 5. Evaluăm modelul pe setul de test
y_pred = tree_reg.predict(X_test)
# - .predict() returnează predicțiile pentru fiecare rând din X_test

r2 = r2_score(y_test, y_pred) # R²: proporția varianței explicată
mse = mean_squared_error(y_test, y_pred) # MSE: eroarea pătratică medie
mae = mean_absolute_error(y_test, y_pred) # MAE: eroarea absolută medie

print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")

# Inspectăm importanța caracteristicilor în deciziile arborelui
print("Feature importances:", tree_reg.feature_importances_)
# - tree_reg.feature_importances_ e un array cu ponderea fiecărei caracteristici
# - valorile sumate sunt 1; valori mai mari = caracteristică mai importantă

# (Grafica arborelui de decizie a fost omisă la cerere)
