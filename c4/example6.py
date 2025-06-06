import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

# 1. Încărcăm o imagine de probă și normalizăm valorile pixelilor în intervalul [0, 1]
sample_image = load_sample_image("china.jpg")
#- load_sample_image încarcă o imagine de test din sklearn
# - image are forma (înălțime, lățime, canale RGB)
data = np.array(sample_image, dtype=np.float64) / 255.0
# - Convertim la float64 și împărțim la 255 pentru a obține valori între 0 și 1
h, w, c = data.shape
# - h: înălțime, w: lățime, c: numărul de canale (RGB are 3)

# 2. Remodelăm imaginea într-un array 2D de pixeli: (h * w, canale)
# - Vrem o matrice unde fiecare rând este un pixel RGB
data_reshaped = np.reshape(data, (h * w, c))
# - data_reshaped are forma (număr_total_pixeli, 3)

# 3. Aplicăm K-Means pentru a reduce paleta de culori
n_colors = 16  # Numărul de culori dorit în paleta comprimată
kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(data_reshaped)
# - .fit() antrenează KMeans pe toți pixelii sub formă de puncte în 3D (R, G, B)
labels = kmeans.predict(data_reshaped)
# - .predict() atribuie fiecărui pixel eticheta clusterului corespunzător culorii

# 4. Reconstruim imaginea comprimată folosind centroizii clusterelor
compressed_data = kmeans.cluster_centers_[labels]
# - Pentru fiecare pixel, înlocuim culoarea originală cu culoarea centroidului clusterului său
compressed_data = np.reshape(compressed_data, (h, w, c))
# - Remodelăm înapoi la forma (înălțime, lățime, canale) pentru a obține imaginea finală

# 5. Afișăm originalul și imaginea comprimată una lângă alta
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(data) # Afișăm imaginea originală (valori în [0,1])
plt.axis('off') # Ascundem axele

plt.subplot(1, 2, 2)
plt.title(f"Compressed ({n_colors} colors)")
plt.imshow(compressed_data)  # Afișăm imaginea comprimată
plt.axis('off')

plt.tight_layout()
plt.show()
# - Imaginea din dreapta are doar 16 culori distincte în loc de milioane,
# ceea ce reduce semnificativ spațiul de stocare, cu pierdere minimă vizibilă.
