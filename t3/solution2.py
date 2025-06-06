import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Creăm un set sintetic non-liniar
np.random.seed(42) # Asigurăm reproducibilitatea zgomotului
num_samples = 30

# Generăm o singură caracteristică X (de exemplu, "Feature")
X = np.linspace(0, 10, num_samples).reshape(-1, 1)
# X are forma (30, 1) – 30 de valori uniform spațiate între 0 și 10
x_flat = X.flatten()
# x_flat are forma (30,) pentru a calcula relația adevărată

# Relația adevărată: y = 2 * X^2 - 3 * X + zgomot
y_true = 2 * (x_flat**2) - 3 * x_flat
noise = np.random.normal(0, 3, size=num_samples) # Zgomot normal cu sigma=3
y = y_true + noise # Observații cu zgomot

# Convertim într-un DataFrame
df = pd.DataFrame({"Feature": x_flat, "Target": y})
# DataFrame-ul are două coloane: 'Feature' și 'Target'

# 2. Separăm caracteristica (X) de țintă (y)
X = df[["Feature"]] # DataFrame cu o coloană pentru model
y = df["Target"] # Serie Pandas cu valorile țintă

# 3. Împărțim în seturi de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# test_size=0.3 → 30% pentru test, random_state=42 pentru reproducibilitate

# 4. Transformăm caracteristica în termeni polinomiali (gradul 2)
poly_degree = 2
poly = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly.fit_transform(X_train)
# fit_transform produce coloanele [1, X, X^2]
X_test_poly = poly.transform(X_test)
# transform păstrează aceeași structură polinomială pentru datele de test

# 5. Creăm și antrenăm modelul de regresie liniară pe caracteristici polinomiale
model = LinearRegression()
model.fit(X_train_poly, y_train)
# Modelul învață coeficienții pentru [1, X, X^2]

# 6. Evaluăm modelul pe setul de test
y_pred = model.predict(X_test_poly)
# Predicții pentru datele de test

r2 = r2_score(y_test, y_pred) # R²: cât de multă varianță explică modelul
mse = mean_squared_error(y_test, y_pred) # MSE: media pătratelor erorilor
mae = mean_absolute_error(y_test, y_pred) # MAE: media erorilor absolute

print("Polynomial Degree:", poly_degree)
print("Coefficients:", model.coef_)  # Coeficienții [coef_intercept, coef_X, coef_X^2]
print("Intercept:", model.intercept_) # Intercept (b0)
print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")

# 7. Plot pentru a vizualiza potrivirea polinomială
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
# Generăm un grid de 100 puncte între 0 și 10
X_range_poly = poly.transform(X_range)
# Transformăm grid-ul la [1, X, X^2]
y_range_pred = model.predict(X_range_poly)
# Predicții pentru grid

plt.scatter(X, y, label="Data")  
# Punctele observate (Feature vs Target)
plt.plot(X_range, y_range_pred, color="red", label="Polynomial Fit")
# Curba de potrivire polinomială
plt.title("Polynomial Regression Example")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()