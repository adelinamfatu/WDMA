import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Presupunem că `df_scaled` este DataFrame-ul preprocesat din Exercitiul 1
# În acest exemplu, simulăm `df_scaled` cu datele Iris scalate:
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 2. Realizăm clustering aglomerativ (ward linkage, 3 clustere)
agg = AgglomerativeClustering(
    n_clusters=3,  # dorim 3 clustere
    linkage='ward' # metoda Ward minimizează varianța în interiorul clusterelor
)
labels = agg.fit_predict(df_scaled)

# 3. Adăugăm etichetele de cluster la DataFrame
df_scaled['cluster'] = labels

# 4. Afișăm câte puncte au fost atribuite fiecărui cluster
print(df_scaled['cluster'].value_counts())

# 5. Construim matricea de legături (linkage) pentru dendrogramă
# Excludem coloana 'cluster' când calculăm linkage
Z = linkage(
    df_scaled.drop(columns=['cluster']), # doar caracteristici, fără etichete
    method='ward', # aceeași metodă Ward folosită la AgglomerativeClustering
    metric='euclidean' # distanța euclidiană între vectorii de caracteristici
)

# 6. Plotăm dendrograma
plt.figure(figsize=(8, 6))
dendrogram(
    Z,
    labels=[f"Obs_{i}" for i in range(df_scaled.shape[0])]  # etichete personalizate pentru fiecare observație
)
plt.title("Dendrogram (Ward linkage) pentru Iris (preprocesat)")
plt.xlabel("Observații")
plt.ylabel("Distanță de legătură")
plt.show()
