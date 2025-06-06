import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generăm date sintetice despre clienți
# - 'purchase_frequency': numărul de achiziții pe lună (1–14)
# - 'average_spent': suma medie cheltuită per achiziție (10–499)
# - 'loyalty_score': scor de loialitate (1–5)
np.random.seed(42)  # pentru reproducibilitate
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),
    'average_spent':       np.random.randint(10, 500, num_customers),
    'loyalty_score':       np.random.randint(1, 6, num_customers)
})
print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scalăm datele
# - StandardScaler transformă fiecare caracteristică astfel încât să aibă media = 0 și deviația standard = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_customers)

# 3. Aplicăm K-Means clustering
# - Alegem, de exemplu, 3 segmente (clustere) pentru clienți
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 4. Adăugăm etichetele de cluster la DataFrame
df_customers['segment'] = kmeans.labels_
# - kmeans.labels_ este un array de lungime 50 cu valorile {0, 1, 2}, indicând segmentul fiecărui client

# 5. Inspectăm fiecare segment
# - Putem calcula media caracteristicilor pentru fiecare segment
segment_summary = df_customers.groupby('segment').mean()
print("=== Segment Summary (mean values) ===")
print(segment_summary, "\n")

# 6. (Opțional) Interpretare rapidă sau acțiuni de marketing
#    - De exemplu, clusterul 0 s-ar putea numi "clienți frecvenți și cu cheltuieli mari"
#    - clusterul 1 s-ar putea numi "clienți ocazionali cu cheltuieli mici" etc.