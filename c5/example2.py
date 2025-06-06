import numpy as np

# Generăm date simple de antrenament (puncte 2D)
X = np.array([
    [0.2, 0.6],
    [0.5, 0.4],
    [0.8, 0.9],
    [0.3, 0.2],
    [0.1, 0.7],
    [0.6, 0.5]
])
# - X.shape = (6, 2): 6 mostre, fiecare cu 2 caracteristici (x, y)

# Etichete: 1 dacă y > x, altfel 0
y = np.array([
    1,  # 0.6 > 0.2
    0,  # 0.4 ≤ 0.5
    1,  # 0.9 > 0.8
    0,  # 0.2 ≤ 0.3
    1,  # 0.7 > 0.1
    0   # 0.5 ≤ 0.6
])
# - y.shape = (6,)

# Adăugăm termenul de bias (coloană de 1)
# - Vom avea acum 3 intrări: x, y și bias=1
X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
# - X_bias.shape = (6, 3)

# Inițializăm greutățile random (pentru cei 3 coeficienți: w_x, w_y și bias)
weights = np.random.randn(3)
# - weights este un vector de dimensiune 3, cu valori din distribuție normală

# Rata de învățare și numărul de epoci
lr = 0.1
epochs = 20

# Funcția de activare: funcție step (trecere) care returnează 1 dacă argumentul ≥ 0, altfel 0
def step(x):
    return 1 if x >= 0 else 0

# Bucla de antrenament
for epoch in range(epochs):
    total_errors = 0
    # Pentru fiecare mostră (xi) și etichetă țintă (target):
    for xi, target in zip(X_bias, y):
        # Calculăm produsul scalar între intrări (inclusiv bias) și greutăți
        z = np.dot(xi, weights)
        # Apelăm funcția step pentru a obține predicția (0 sau 1)
        pred = step(z)
        # Calculăm eroarea simplă: diferența dintre eticheta țintă și predicție
        error = target - pred
        # Actualizăm greutățile: w ← w + lr * error * xi
        weights += lr * error * xi
        # Incrementăm numărul total de erori (valorile absolute ale eroarelor)
        total_errors += abs(error)
    # Afișăm numărul de erori pentru epoca curentă
    print(f"Epoch {epoch+1}: Errors = {total_errors}")

# Funcție de predicție pentru un nou punct (fără încărcare de buclă)
def predict(x_point):
    # Adăugăm bias (1) la vectorul de caracteristici
    x_with_bias = np.append(x_point, 1)
    # Calculăm produsul scalar și aplicăm funcția step
    return step(np.dot(x_with_bias, weights))

# Testăm pe un punct nou
test_point = [0.4, 0.7]
# - 0.7 > 0.4, așteptăm clasificare 1
print(f"Point {test_point} classified as: {predict(test_point)}")
