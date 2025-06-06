import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Generăm un set mic sintetic de date
# - Folosim 2 caracteristici numerice pentru simplitate
np.random.seed(42)  # Pentru reproducibilitate
num_samples = 20
X = np.random.rand(num_samples, 2) * 100  
# - X are forma (20, 2), valorile între 0 și 100 pentru fiecare caracteristică
# - Considerăm relația adevărată price = 3*Feature1 + 2*Feature2 + zgomot
true_coeffs = np.array([3.0, 2.0])
y = X.dot(true_coeffs) + np.random.normal(0, 10, size=num_samples)
# - Adăugăm zgomot normal cu sigma=10

# Convertim într-un DataFrame pentru familiaritate
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Price"] = y
# - DataFrame-ul conține trei coloane: Feature1, Feature2 și Price (țintă)

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[["Feature1", "Feature2"]]
y = df["Price"]
# - X este DataFrame-ul cu cele două coloane de intrare
# - y este Serie cu prețurile generate

# 3. Împărțim setul în antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Creăm și antrenăm un model de regresie liniară
model = LinearRegression()
model.fit(X_train, y_train)
# - .fit() estimează coeficienții (pantele) și interceptul care minimizează eroarea pătratică

# 5. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - .predict() aplică ecuația învățată pentru date noi

# 6. Evaluăm modelul folosind R², MSE și MAE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# - R² măsoară cât de multă varianță explică modelul (1.0 = potrivire perfectă)
# - MSE penalizează pătratele erorilor (diferență actual-prezis)
# - MAE este media valorilor absolute ale erorilor

print("Coefficients:", model.coef_)
# - model.coef_ afișează pantele estimate pentru Feature1 și Feature2
print("Intercept:", model.intercept_)
# - model.intercept_ este valoarea estimată când Feature1=Feature2=0
print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")
