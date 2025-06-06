import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generăm date sintetice
# - Presupunem că adevărata relație între viteză și distanța de frânare este un polinom de gradul 2,
# iar în realitate adăugăm și un pic de zgomot (noise).
np.random.seed(42) # Asigurăm reproducibilitatea secvenței de „random”
speeds = np.linspace(10, 100, 20) # 20 de valori ale vitezei între 10 și 100 km/h

# - Calculăm „adevărata” distanță de frânare ca 0.02 * speed^2 - 1.5 * speed + 50
true_braking_distance = 0.02 * speeds**2 - 1.5 * speeds + 50

# - Generăm zgomot normal (mean=0, std=20) cu aceeași dimensiune ca speeds
noise = np.random.normal(loc=0.0, scale=20.0, size=len(speeds))

# - Distanța observată = valoarea adevărată + zgomot
braking_distance = true_braking_distance + noise

# 2. Pregătim datele pentru scikit-learn și aplicăm transformarea polinomială
# - scikit-learn așteaptă matrice 2D pentru caracteristici, așa că reshăpăm
X = speeds.reshape(-1, 1) # din (20,) în (20, 1)

# - PolynomialFeatures(degree=2) creează caracteristici: [1, x, x^2]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# - X_poly va avea 3 coloane: 
# [1 (intercept), speed, speed^2] pentru fiecare eșantion

# 3. Antrenăm un model de regresie liniară pe noile caracteristici polinomiale
model = LinearRegression()
model.fit(X_poly, braking_distance)
# - Acum modelul învață coeficienți pentru termenii 1, x și x^2

# 4. Pregătim puncte noi, „netede”, pentru a desena curba polinomială a modelului
# - speeds_plot este un vector de 100 de puncte între min(speeds) și max(speeds)
speeds_plot = np.linspace(min(speeds), max(speeds), 100).reshape(-1, 1)

# - Aplicăm aceeași transformare polinomială (generăm [1, x, x^2] pentru fiecare nou punct)
speeds_plot_poly = poly.transform(speeds_plot)

# - Obținem predicțiile pentru curba polinomială
braking_distance_pred = model.predict(speeds_plot_poly)

# 5. Plotăm datele observate și curba ajustată de polinom
plt.scatter(speeds, braking_distance, label="Data (Observed)")
# - scatter afișează punctele reale (viteza vs. distanță cu zgomot)
plt.plot(speeds_plot, braking_distance_pred, color='red', label="Polynomial Fit")
# - plot afișează curba „suavizată” a modelului polinomial
plt.xlabel("Car's Speed (km/h)")
plt.ylabel("Braking Distance (m)")
plt.title("Polynomial Regression: Speed vs. Braking Distance")
plt.legend()
plt.show()
# - Afișează fereastra cu graficul
