import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generăm date sintetice care reprezintă trafic normal
# - Fiecare punct are două caracteristici (ex.: avg_packet_size, connection_speed)
np.random.seed(42)  
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))
# - normal_data shape = (200, 2), valorile sunt distribuite normal cu medie=50 și sigma=10

# 2. Generăm date sintetice care reprezintă trafic neobișnuit (anomalie)
# - Puncte mai dispersate, provenind dintr-o distribuție uniformă pe intervalul [0,100]
outliers = np.random.uniform(low=0, high=100, size=(10, 2))
# - outliers shape = (10, 2), valorile uniforme între 0 și 100

# 3. Combinăm datele normale și pe cele de anomalie
X = np.vstack((normal_data, outliers))
# - X shape = (210, 2), primii 200 sunt date normale, ultimii 10 sunt potențiale anomalii

# 4. Aplicăm DBSCAN pentru detectarea anomaliilor
# - eps = 3 definește raza de vecinătate (în unități ale caracteristicilor)
# - min_samples = 5 înseamnă că cel puțin 5 puncte trebuie să fie în raza eps pentru a forma un „cluster dens”
dbscan = DBSCAN(eps=3, min_samples=5)
labels = dbscan.fit_predict(X)
# - labels este un array de lungime 210:
# * valorile >= 0 indică indexul clusterului atribuit fiecărui punct
# * valoarea -1 indică un punct considerat outlier (anomalie)

# 5. Identificăm punctele etichetate ca anomalii (labels == -1) și cele normale
outlier_points = X[labels == -1]
normal_points = X[labels != -1]
# - outlier_points shape = (numarul de puncte cu label -1, 2)
# - normal_points shape = (restul punctelor, 2)

# 6. Vizualizăm rezultatele
# - Punctele normale sunt colorate după eticheta clusterului
# - Anomaliile (outliers) sunt marcate cu 'x' roșu
plt.scatter(normal_points[:, 0], normal_points[:, 1], c=labels[labels != -1], cmap='viridis', s=50)
# - normal_points[:, 0] și normal_points[:, 1] sunt coordonatele punctelor normale
# - c=labels[...] colorează fiecare punct normal în funcție de clusterul său
plt.scatter(outlier_points[:, 0], outlier_points[:, 1], marker='x', s=100, color='red', label='Anomaly')
# - outlier_points[:, 0] și outlier_points[:, 1] sunt coordonatele punctelor detectate ca anomalii
plt.title("Network Traffic Anomaly Detection with DBSCAN")
plt.xlabel("Feature 1 (e.g. Avg Packet Size)")
plt.ylabel("Feature 2 (e.g. Connection Speed)")
plt.legend()
plt.show()
# - Anomaliile apar ca 'x' roșu, separate de punctele grupate în clustere

# 7. Afișăm un sumar al anomaliilor detectate
print("Number of anomalies detected:", len(outlier_points))
print("Sample anomaly points:\n", outlier_points[:5])
# - len(outlier_points) arată câte puncte au fost marcate ca outlieri
# - Afișăm primele 5 pentru inspecție