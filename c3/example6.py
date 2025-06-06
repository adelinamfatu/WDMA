import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# 1. Generăm un set mic sintetic de date
np.random.seed(42)  # pentru reproducibilitate

# Caracteristicile X: 100 de rânduri și 5 coloane cu valori între 0 și 10
X = np.random.rand(100, 5) * 10
# Coeficienții „adevărați” ai modelului (unii sunt zero pentru a demonstra efectul L1)
true_coefs = np.array([1.5, 0.0, -2.0, 0.0, 3.0])
# Ținta y = X · true_coefs + zgomot normal cu sigma = 2
y = X.dot(true_coefs) + np.random.normal(0, 2, size=100)

# 2. Împărțim în set de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Antrenăm un model Ridge (regularizare L2, alpha=1.0)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_coefs = ridge.coef_ # coeficienții obținuți de Ridge
ridge_intercept = ridge.intercept_

# 4. Antrenăm un model Lasso (regularizare L1, alpha=1.0)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_coefs = lasso.coef_ # coeficienții obținuți de Lasso (unii pot fi exact 0)
lasso_intercept = lasso.intercept_

# 5. Comparăm coeficienții „adevărați” cu cei estimați de Ridge și Lasso
print("True coefficients:", true_coefs)
print("\nRidge coefficients:", ridge_coefs)
print("Ridge intercept:", ridge_intercept)
print("\nLasso coefficients:", lasso_coefs)
print("Lasso intercept:", lasso_intercept)

# 6. (Opțional) Evaluăm R^2 pe setul de test pentru ambele modele
ridge_score = ridge.score(X_test, y_test)
lasso_score = lasso.score(X_test, y_test)
print(f"\nRidge R^2 on test data: {ridge_score:.3f}")
print(f"Lasso R^2 on test data: {lasso_score:.3f}")
