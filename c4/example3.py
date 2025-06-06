import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Generăm vectori de caracteristici sintetici
# - Fiecare rând reprezintă descriptorul unei imagini (de ex. un vector de 64 dimensiuni)
np.random.seed(42)
num_images = 10
num_features = 64
image_features = np.random.rand(num_images, num_features)
# - image_features este un array cu forma (10, 64), conținând valori aleatoare între 0 și 1

# 2. Realizăm clustering ierarhic (linkage)
# - Alegem metoda "ward", care minimizează varianța totală în cadrul fiecărui cluster
# - Alte opțiuni ar fi "single", "complete" sau "average"
Z = linkage(image_features, method='ward')
# - Z este o matrice (n-1, 4) care codifică pașii de îmbinare a clusterelor:
# * primele două coloane sunt indecșii clusterelor îmbinate
# * a treia coloană este "distanța" la care s-au unit (lungimea ramurii)
# * a patra coloană este numărul de puncte din noul cluster

# 3. Plotăm dendrograma
plt.figure(figsize=(8, 6))
dendrogram(
    Z,
    labels=[f"Img_{i}" for i in range(num_images)]  # etichete prietenoase pentru fiecare imagine
)
plt.title("Dendrogram of Synthetic Image Feature Vectors")
plt.xlabel("Images")
plt.ylabel("Distance (Ward linkage)")
plt.show()
# - Fiecare frunză (leaf) corespunde unei imagini (Img_0, Img_1, ..., Img_9)
# - Ramurile arată ordinea și distanțele la care se unesc clusterele
# - Lungimea verticală a ramurii indică cât de „distanțate” erau cele două grupuri înainte de a fi unite

# 4. Cum interpretăm dendrograma:
# - Fiecare etichetă de pe axa orizontală este o imagine distinctă.
# - Pe măsură ce ne ridicăm de la nivelul frunză spre rădăcină, vedem cum se unesc imagini similare în clustere din ce în ce mai mari.
# - Dacă trasăm o linie orizontală la o anumită distanță (de ex. 7), taind dendrograma, vom obține un număr concret de clustere (acolo unde linia intersectează ramurile).
# - Ramurile scurte (distanțe mici) indică imagini foarte asemănătoare în caracteristici, iar ramurile foarte lungi (distanțe mari) semnalează că două grupuri erau foarte diferite înainte de îmbinare.
