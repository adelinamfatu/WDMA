import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. Încărcăm setul de date MNIST (70000 eșantioane, fiecare 28x28 pixeli -> 784 caracteristici)
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# - X are forma (70000, 784), valorile pixelilor între 0 și 255
X = X / 255.0  # Normalizăm valorile pixelilor în intervalul [0, 1]
# - Convertim la float și împărțim la 255 pentru standardizare
y = y.astype(int)  # Convertim etichetele la tip int

# 2. One-hot encode pentru etichete (transformăm valorile 0-9 în vectori de dimensiune 10)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.to_numpy().reshape(-1, 1))
# - fit_transform produce un array shape (70000, 10), cu un '1' în poziția clasei corecte

# 3. Folosim un subset de 10000 eșantioane pentru viteză (evităm să antrenăm pe toate cele 70000)
X_train, _, y_train, _ = train_test_split(
    X, y_onehot,
    train_size=10000,
    random_state=42
)
# - X_train shape = (10000, 784), y_train shape = (10000, 10)

X_train = X_train.to_numpy()  # Convertim la array NumPy pentru comoditate
y_train = np.asarray(y_train)  # Asigurăm că y_train e un array NumPy

# 4. Setăm mărimile pentru stratul de intrare, ascuns și ieșire
input_size = 784    # fiecare imagine are 784 pixeli
hidden_size = 64    # vom folosi 64 de neuroni în stratul ascuns
output_size = 10    # 10 clase (cifre de la 0 la 9)
lr = 0.1            # rata de învățare
epochs = 10         # numărul de epoci de antrenament
batch_size = 64     # mărimea fiecărui batch

# 5. Inițializăm greutățile și bias-urile
W1 = np.random.randn(input_size, hidden_size) * 0.01
#    - matricea de greutăți pentru stratul 1 (784 x 64), valori mici random
b1 = np.zeros((1, hidden_size))
#    - vectorul de bias-uri pentru stratul 1 (1 x 64), inițial zero

W2 = np.random.randn(hidden_size, output_size) * 0.01
#    - matricea de greutăți pentru stratul 2 (64 x 10), valori mici random
b2 = np.zeros((1, output_size))
#    - vectorul de bias-uri pentru stratul 2 (1 x 10), inițial zero

# 6. Definim funcțiile de activare și derivata lor
def relu(x):
    return np.maximum(0, x)
    #    - ReLU: păstrează valorile pozitive, pune zero la negative

def relu_derivative(x):
    return (x > 0).astype(float)
    #    - Derivata ReLU: 1 acolo unde x > 0, 0 altundeva

def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    #    - Scădem maximul pe fiecare rând pentru stabilitate numerică
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    #    - Fiecare rând devine probabilități care însumează 1

# 7. Funcția de pierdere Cross-Entropy
def cross_entropy(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions + 1e-8), axis=1))
    #    - Adăugăm un mic 1e-8 pentru a evita log(0)

# 8. Bucla de antrenament
for epoch in range(epochs):
    # 8.1. Amestecăm datele la fiecare epocă
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    # 8.2. Parcurgem fiecare batch
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # -------- Forward pass --------
        z1 = X_batch @ W1 + b1      # (batch_size x 784) @ (784 x 64) → (batch_size x 64)
        a1 = relu(z1)               # aplicăm ReLU la z1
        z2 = a1 @ W2 + b2           # (batch_size x 64) @ (64 x 10) → (batch_size x 10)
        a2 = softmax(z2)            # aplicăm softmax, obținem probabilități pentru cele 10 clase

        # Calculăm pierderea (pentru primul batch din epocă, ca exemplu)
        if i == 0:
            loss = cross_entropy(a2, y_batch)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # -------- Backward pass (calculul gradientului) --------
        dz2 = a2 - y_batch
        #    - gradienții pentru stratul 2: (batch_size x 10)

        dW2 = a1.T @ dz2 / batch_size
        #    - matricea de gradient pentru W2: (64 x batch_size) @ (batch_size x 10) → (64 x 10)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        #    - gradientul pentru b2: sumăm pe fiecare coloană → (1 x 10)

        da1 = dz2 @ W2.T
        #    - gradient pentru activările stratului 1: (batch_size x 10) @ (10 x 64) → (batch_size x 64)
        dz1 = da1 * relu_derivative(z1)
        #    - aplicăm derivata ReLU element-wise pe z1

        dW1 = X_batch.T @ dz1 / batch_size
        #    - gradient pentru W1: (784 x batch_size) @ (batch_size x 64) → (784 x 64)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        #    - gradient pentru b1: sumăm pe fiecare coloană → (1 x 64)

        # -------- Actualizare greutăți (Gradient Descent) --------
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

# 9. Evaluare pe datele de antrenament
z1 = X_train @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
preds = np.argmax(z2, axis=1)
#    - Alegem clasa cu probabilitatea maximă (fără softmax necesar aici, pentru clasificare)

true = np.argmax(y_train, axis=1)
#    - Aflăm eticheta adevărată (indicele elementului '1' din one-hot)

acc = np.mean(preds == true)
print(f"Training accuracy: {acc * 100:.2f}%")
# - Calculăm proporția predicțiilor corecte
