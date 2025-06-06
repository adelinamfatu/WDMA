import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. Creăm un dataset simplu cu cheltuieli de publicitate și vânzări
# - În realitate, s-ar citi dintr-un fișier CSV sau bază de date
data = {
    'TV':        [230.1, 44.5, 17.2, 151.5, 180.8, 8.7,   57.5,  120.2, 8.6,   199.8],
    'Radio':     [37.8,  39.3,  45.9, 41.3,  10.8,  48.9,  32.8,  19.6,  2.1,   2.6],
    'Newspaper': [69.2,  45.1,  69.3, 58.5,  58.4,  75.0,  23.5,  11.6,  1.0,   21.2],
    'Sales':     [22.1,  10.4,  9.3,  18.5,  12.9,  7.2,   11.8,  13.2,  4.8,   10.6]
}
df = pd.DataFrame(data)
# - df este un DataFrame cu 4 coloane:
# 'TV', 'Radio', 'Newspaper' (date de intrare) și 'Sales' (țintă)

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[['TV', 'Radio', 'Newspaper']] # DataFrame cu cele trei coloane de publicitate
y = df['Sales'] # Serie cu valorile vânzărilor

# 3. Împărțim datele în set de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# - test_size=0.2 înseamnă că 20% din date vor fi folosite pentru evaluare
# - random_state=42 asigură aceeași împărțire aleatorie la fiecare rulare

# 4. Creăm și antrenăm modelul de regresie liniară
model = LinearRegression()
model.fit(X_train, y_train)
# - .fit() găsește coeficienții (pantele) și interceptul care minimizează eroarea pătratică

# 5. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - .predict() aplică ecuația învățată pentru a genera predicțiile de vânzări

# 6. Evaluăm modelul folosind R² și MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# - R² indică proporția varianței vânzărilor explicată de model (1 = potrivire perfectă)
# - MSE = media pătratelor diferențelor (actual vs. prezis), penalizând erorile mari

print("Coefficients (TV, Radio, Newspaper):", model.coef_)
# - model.coef_ este un array cu câte un coeficient pentru fiecare caracteristică (TV, Radio, Newspaper)
print("Intercept:", model.intercept_)
# - model.intercept_ este valoarea de pornire (când toate caracteristicile sunt 0)
print("R^2 on test set:", r2)
print("MSE on test set:", mse)
