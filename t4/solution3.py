import pandas as pd
from sklearn.cluster import DBSCAN

# 1. Presupunem că `df_scaled` este DataFrame-ul preprocesat din Exercitiul 1
# Pentru demonstrație, simulăm `df_scaled` cu datele Iris scalate
from sklearn.datasets import load_iris
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# - În situația reală, înlocuiește acest bloc cu DataFrame-ul tău scalat

# 2. Instanțiem DBSCAN cu parametri aleși
# - eps definește raza de vecinătate (similaritate maximă)
# - min_samples este numărul minim de puncte pentru ca o regiune să fie considerată "densă"
dbscan = DBSCAN(eps=0.5, min_samples=5)
# - eps=0.5: orice două puncte mai aproape de 0.5 vor fi considerate vecine
# - min_samples=5: cel puțin 5 puncte în raza eps pentru a forma un cluster dens

# 3. Potrivim (fit) modelul pe datele scalate
dbscan.fit(df_scaled)

# 4. Extragem etichetele de cluster
labels = dbscan.labels_

# 5. Identificăm outlierii (cei cu label == -1)
outliers = df_scaled[labels == -1]

# 6. (Opțional) Adăugăm etichetele DBSCAN la DataFrame
df_scaled['cluster'] = labels

# 7. Afișăm numărul de apariții pentru fiecare etichetă de cluster
print(df_scaled['cluster'].value_counts())

# 8. (Comentariu) Vizualizarea rapidă a clusterelor pentru două caracteristici
#    - Dacă vrei să afișezi un scatter plot, alege două coloane și colorează după df_scaled['cluster']
#    - Exemplu (fără afișat):
#      plt.scatter(df_scaled['sepal length (cm)'], df_scaled['sepal width (cm)'], c=df_scaled['cluster'])
#      plt.show()
