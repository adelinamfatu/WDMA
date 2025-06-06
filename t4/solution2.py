import pandas as pd
from sklearn.cluster import KMeans

# 1. Presupunem că `df_scaled` este DataFrame-ul preprocesat din Exercitiul 1
# (conține caracteristici numerice, imputate și scalate).
# Aici simulăm `df_scaled` cu datele Iris scalate (pentru demonstrație).
from sklearn.datasets import load_iris
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# - În exercițiul real, înlocuiește blocul de simulare cu DataFrame-ul tău scalat

# 2. Instanțiem K-Means cu un număr de clustere ales, de exemplu 3
kmeans = KMeans(n_clusters=3, random_state=42)
# - n_clusters=3: vom împărți datele în 3 grupuri (clustere)
# - random_state=42: asigură aceeași poziționare inițială a centroidelor la fiecare rulare

# 3. Potrivim (fit) modelul pe datele scalate
kmeans.fit(df_scaled)
# - .fit() găsește centroizii optimi pe baza datelor din df_scaled

# 4. Extragem etichetele de cluster (0, 1 sau 2 pentru fiecare rând)
labels = kmeans.labels_
# - labels este un array de lungime egală cu numărul de rânduri din df_scaled
# - Fiecare valoare indică indicelui clusterului corespunzător

# 5. (Opțional) Adăugăm etichetele la DataFrame pentru a păstra segmentarea
df_scaled['cluster'] = labels
# - Adăugăm o coloană 'cluster' în df_scaled cu valorile din labels
# - Astfel, putem analiza ulterior fiecare cluster separat

# 6. Afișăm primele câteva rânduri pentru a vedea etichetele de cluster
print(df_scaled.head(10))
# - Acum putem observa care rânduri (clienți/observații) au fost atribuite fiecărui cluster
