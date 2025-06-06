import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generăm un set mic de date cu zgomot (numai 8 puncte)
np.random.seed(0)  # Asigurăm reproducibilitatea zgomotului
X_small = np.linspace(-3, 3, 8) # 8 valori uniform spațiate între -3 și 3
y_true = 0.5 * X_small**2 # Trendul “real” subiacente: y = 0.5 * x^2 (quadratic)
noise = np.random.normal(loc=0.0, scale=2.0, size=len(X_small))  
# - Generăm zgomot normal cu media 0 și deviația standard 2, pentru fiecare punct
y_small = y_true + noise # Datele observate = trendul real + zgomot

# Reshapăm X_small într-un array 2D (forma necesară de scikit-learn: (n_samples, 1))
X_small = X_small.reshape(-1, 1)

# 2. Creăm caracteristici polinomiale de grad înalt (degree=9)
poly = PolynomialFeatures(degree=9)
X_poly = poly.fit_transform(X_small)
#- Pentru fiecare x în X_small, poly.fit_transform generează [1, x, x^2, x^3, ..., x^9]
# - Dimensiunea devine (8, 10): 1 coloană pentru intercept și 9 coloane pentru puterile lui x

# 3. Antrenăm un model liniar pe aceste caracteristici polinomiale
model = LinearRegression()
model.fit(X_poly, y_small)
# - Chiar dacă datele includ x^2, x^3, etc., algoritmul de regresie rămâne “liniar”
# - Învață coeficienți pentru fiecare termen din [1, x, x^2, …, x^9]

# 4. Pregătim un set de puncte “fine” pentru afișarea curbei
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1) # 200 de puncte între -3 și 3
X_plot_poly = poly.transform(X_plot) # Generăm [1, x, x^2, …, x^9] pentru fiecare dintre cele 200 puncte
y_plot_pred = model.predict(X_plot_poly) # Folosim modelul antrenat pentru a prezice y pentru aceste puncte

# 5. Plotăm datele și potrivirea înalt polinomială
plt.scatter(X_small, y_small, label="Noisy Data Points")  
# - Afișăm cele 8 puncte observate (zgomotoase) ca scatter
plt.plot(X_plot, y_plot_pred, label="Degree=9 Polynomial Fit")  
# - Afișăm curba de potrivire (modelul polinomial de grad 9) pe un grid fin de 200 puncte
plt.plot(X_plot, 0.5 * X_plot**2, label="True Underlying Trend (Quadratic)")  
# - Afișăm și “linia” reală fără zgomot: y = 0.5 x^2
plt.title("High-Degree Polynomial Overfitting Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
# - Grafical, vedem că polinomul de grad 9 “îmbrățișează” toate punctele zgomotoase (overfitting),
# în timp ce trendul real este doar o parabolă.
