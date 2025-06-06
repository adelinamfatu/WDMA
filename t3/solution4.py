import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Generăm un set de date sintetic cu 3 caracteristici
np.random.seed(42)
num_samples = 30

# Caracteristicile:
# X1 și X2 sunt corelate (X2 = X1 + puțin zgomot)
# X3 este aleatorie, posibil mai puțin relevantă
X1 = np.random.rand(num_samples) * 10
X2 = X1 + np.random.rand(num_samples) * 2
X3 = np.random.rand(num_samples) * 10

# Ținta 'y' – relația "adevărată":
# y = 3*X1 + 1.5*X2 + zgomot
y = 3 * X1 + 1.5 * X2 + np.random.normal(0, 5, size=num_samples)

# Convertim într-un DataFrame pentru claritate
df = pd.DataFrame({
    "X1": X1,
    "X2": X2,
    "X3": X3,
    "Target": y
})
# - 'X1', 'X2', 'X3' sunt coloane de caracteristici
# - 'Target' este coloana țintă (y)

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[["X1", "X2", "X3"]]  # caracteristici
y = df["Target"]           # țintă

# 3. Împărțim în seturi de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Definim modelele Ridge și Lasso (alpha=1.0)
ridge = Ridge(alpha=1.0) # penalizare L2
lasso = Lasso(alpha=1.0) # penalizare L1 (selecție variabile)

# Antrenăm modelele pe datele de antrenament
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
# - .fit() estimează coeficienții (pantele) și interceptul cu regularizarea specificată

# 5. Evaluăm modelul Ridge pe setul de test
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge) # R²: proporția varianței explicată
mse_ridge = mean_squared_error(y_test, y_pred_ridge) # MSE: media pătratelor erorilor
mae_ridge = mean_absolute_error(y_test, y_pred_ridge) # MAE: media erorilor absolute

# Evaluăm modelul Lasso pe setul de test
y_pred_lasso = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

# 6. Comparăm coeficienții și performanța fiecărui model
print("True Relationship: y = 3*X1 + 1.5*X2 + noise")

print("\nRidge Coefficients:", ridge.coef_)
print("Ridge Intercept:", ridge.intercept_)
print(f"Ridge R²: {r2_ridge:.3f}, MSE: {mse_ridge:.3f}, MAE: {mae_ridge:.3f}")

print("\nLasso Coefficients:", lasso.coef_)
print("Lasso Intercept:", lasso.intercept_)
print(f"Lasso R²: {r2_lasso:.3f}, MSE: {mse_lasso:.3f}, MAE: {mae_lasso:.3f}")
