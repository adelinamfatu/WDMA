import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Încărcăm și scalăm setul de date Iris
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# - StandardScaler ajustează fiecare caracteristică să aibă medie 0 și deviație standard 1
# - Acest pas e important pentru KMeans, deoarece distanța euclidiană e sensibilă la scări diferite

# 2. Definim intervalul de valori k pe care vrem să le testăm (1 până la 10)
k_values = range(1, 11)

# 3. Pentru fiecare k, antrenăm KMeans și stocăm SSE (inertia)
sse = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    # - n_clusters=k: numărul de clustere încercate
    # - random_state=42: reproducibilitate, pozițiile inițiale ale centroidelor
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    #- inertia_ = Sum of Squared Errors (SSE) față de centroidi
    # - Măsura cât de compact sunt clusterele: valori mai mici înseamnă grupări mai strânse

# 4. Afișăm grafic SSE în funcție de k pentru metoda „cotului” (Elbow Method)
plt.figure(figsize=(6, 4))
plt.plot(k_values, sse, marker='o')
# - marker='o' afișează un punct la fiecare valoare k
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.xticks(k_values)
# - plt.xticks(k_values) asigură afișarea numerelor întregi de la 1 la 10 pe axa X
plt.show()
# - Din acest grafic, căutăm „cotul”, unde scăderea SSE se reduce semnificativ.
