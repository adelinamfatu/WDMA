import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Creăm un set sintetic cu două caracteristici numerice
np.random.seed(42)  
num_samples = 30
X = np.random.rand(num_samples, 2) * 10  
# - X este matricea (30, 2) cu valori între 0 și 10 pentru fiecare caracteristică
# Relația "adevărată" pentru țintă:
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)
# - y = 3*Feature1 + 2*Feature2 + zgomot normal (sigma=5)

# Convertim într-un DataFrame pentru claritate
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y
# - 'Feature1' și 'Feature2' sunt coloanele de intrare
# - 'Target' este coloana cu valorile țintă

# 2. Separăm caracteristicile (X) de țintă (y)
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Împărțim datele în seturi de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Scalare caracteristici (recomandată pentru metode de distanță precum kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# - fit_transform calculează media și deviația standard pe X_train și transformă X_train
X_test_scaled = scaler.transform(X_test)
# - transform aplică aceeași scalare la X_test

# 5. Creăm și antrenăm regressorul kNN cu k=3
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_scaled, y_train)
# - .fit() memorează setul scalat de antrenament; kNN nu are parametri antrenați în sens clasic

# 6. Evaluăm modelul pe setul de test
y_pred = knn_reg.predict(X_test_scaled)
# - .predict() găsește cei 3 cei mai apropiați vecini pentru fiecare exemplu din X_test_scaled și returnează media țintelor lor

r2 = r2_score(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)  
mae = mean_absolute_error(y_test, y_pred)

print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")

# 7. (Opțional) Explorăm efectul alegerii diferitelor valori k
for k in [1, 3, 5, 7]:
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)  
    # - .fit() memorează datele scalate; nu antrenează coeficienți
    y_pred_temp = knn_temp.predict(X_test_scaled)
    # - Predicție pe setul de test
    r2_temp = r2_score(y_test, y_pred_temp)
    print(f"k={k}, R² = {r2_temp:.3f}")
