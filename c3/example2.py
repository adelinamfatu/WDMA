import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Creăm un dataset mic cu o singură caracteristică (ore studiate) și un țintă (scor examen)
hours_studied = np.array([1, 2, 3, 4, 5, 6])
exam_score = np.array([50, 60, 65, 70, 75, 90])

# 2. Reshapăm vectorul de ore studiate într-un array 2D (cerut de scikit-learn)
# - .reshape(-1, 1) face din [1,2,3,...] un [[1],[2],[3],...]
X = hours_studied.reshape(-1, 1)
y = exam_score

# 3. Creăm și antrenăm modelul de regresie liniară
model = LinearRegression()
model.fit(X, y)
# - model.coef_[0] va fi panta liniei
# - model.intercept_ va fi interceptul (punctul unde linia tașă cu axa Y)

# 4. Extragem parametrii modelului
slope = model.coef_[0]   # coeficientul (panta) pentru “ore studiate”
intercept = model.intercept_ # valoarea de pornire când ore_studiate=0
print(f"Slope (Coefficient): {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
# - De exemplu, dacă slope ≈ 7, înseamnă că fiecare oră suplimentară studiată adaugă ~7 puncte în scor

# 5. Facem predicții pe același X pentru a vedea linia de regresie
y_pred = model.predict(X)

# 6. Plotăm punctele reale și linia de best-fit obținută de model
plt.scatter(hours_studied, exam_score, label="Data Points")
plt.plot(hours_studied, y_pred,  label="Best Fit Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Hours Studied vs Exam Score")
plt.legend()
plt.show()
