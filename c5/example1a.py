import tensorflow as tf
import numpy as np

# Convertim datele la array NumPy
X = np.array([
    [0.1, 0.2],
    [0.4, 0.3],
    [0.6, 0.8],
    [0.9, 0.5]
])
# - X.shape = (4, 2): 4 eșantioane, fiecare cu 2 caracteristici

y = np.array([1, 0, 1, 0])
# - y.shape = (4,): etichete binare pentru fiecare eșantion

# Definim modelul secvențial cu două straturi Dense
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        4, 
        activation='relu', 
        input_shape=(2,)
    ),  # strat ascuns cu 4 neuroni și activare ReLU, primește 2 caracteristici
    tf.keras.layers.Dense(
        1, 
        activation='sigmoid'
    )   # strat de ieșire cu 1 neuron și activare Sigmoid (pentru clasificare binară)
])

# Compilăm modelul specificând optimizer-ul și funcția de pierdere
model.compile(
    optimizer='adam', # utilizăm algoritmul Adam pentru actualizarea greutăților
    loss='binary_crossentropy', # pierderea pentru clasificare binară
    metrics=['accuracy'] # vom monitoriza acuratețea în timpul antrenării
)