import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Încărcăm setul de date Wine
wine = load_wine()
X = wine.data # matricea de caracteristici (atributele vinului)
y = wine.target # vectorul țintă (3 clase de vin: 0, 1, 2)

# 2. Împărțim datele în seturi de antrenament și test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% merg în setul de test
    random_state=42 # seed pentru reproducibilitate
)

# 3. Antrenăm un clasificator Naïve Bayes gaussian (potrivit pentru date continue)
model = GaussianNB()
model.fit(X_train, y_train)
# - GaussianNB își calculează media și varianța fiecărei caracteristici pe fiecare clasă
# - la predict, folosește distribuția normală pentru a calcula probabilitățile condiționate

# 4. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - predict returnează eticheta clasă (0, 1 sau 2) pentru fiecare exemplu din X_test

# 5. Afișăm acuratețea modelului pe setul de test
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
# - accuracy_score = proporția de exemple prezise corect (număr corecte / număr total)

# (Opțional) Afișăm matricea de confuzie pentru a vedea distribuția predicțiilor greșite
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_mat)
# - Matricea 3×3: fiecare rând = clasa reală, fiecare coloană = clasa prezisă
# - Ex: conf_mat[i,j] = numărul de exemple din clasa i, prezise ca j
