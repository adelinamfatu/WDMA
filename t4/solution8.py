import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 1. Încărcăm și scalăm setul de date Iris
iris = load_iris()
X = iris.data
y = iris.target # nu îl folosim pentru clustering, dar poate fi util ca reper

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Aplicăm trei algoritmi de clustering pe datele scalate
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Reducem dimensionalitatea la 2D cu PCA pentru vizualizare
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. Construim un DataFrame combinat pentru plot și raport
df_final = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_final['KMeans'] = kmeans.labels_
df_final['DBSCAN'] = dbscan.labels_
df_final['Agglo'] = agg.labels_

# 5. Plotăm diagrame scatter pentru fiecare metodă în spațiul PCA
plt.figure(figsize=(15, 4))

# Plot KMeans
plt.subplot(1, 3, 1)
plt.scatter(df_final['PC1'], df_final['PC2'], c=df_final['KMeans'], cmap='viridis', s=30)
plt.title("K-Means Clustering (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

# Plot DBSCAN
plt.subplot(1, 3, 2)
plt.scatter(df_final['PC1'], df_final['PC2'], c=df_final['DBSCAN'], cmap='viridis', s=30)
plt.title("DBSCAN Clustering (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

# Plot Agglomerative
plt.subplot(1, 3, 3)
plt.scatter(df_final['PC1'], df_final['PC2'], c=df_final['Agglo'], cmap='viridis', s=30)
plt.title("Agglomerative Clustering (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()

# 6. Raport scurt al distribuției clusterelor
print("=== Distribuție clustere KMeans ===")
print(df_final['KMeans'].value_counts().sort_index(), "\n")

print("=== Distribuție clustere DBSCAN ===")
print(df_final['DBSCAN'].value_counts().sort_index(), "\n")

print("=== Distribuție clustere Agglomerative ===")
print(df_final['Agglo'].value_counts().sort_index())
