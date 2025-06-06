import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# 1. Generăm un set de date sintetic pentru clasificare binară
X, y = make_classification(
    n_samples=100, # 100 de exemple
    n_features=5, # 5 caracteristici numerice
    n_informative=3, # 3 dintre ele vor fi cu adevărat relevante pentru țintă
    n_redundant=0, # niciuna nu va fi doar o combinație liniară a celor informative
    n_classes=2, # două clase: 0 și 1
    random_state=42 # pentru reproducibilitate
)

# 2. Împărțim setul în antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 30% dintre exemple merg la test
    random_state=42 # seed fix
)

# 3. Antrenăm un model de regresie logistică pe datele de antrenament
model = LogisticRegression()
model.fit(X_train, y_train)
# - fit învață valorile coeficienților pe baza lui X_train și y_train

# 4. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - predict returnează 0 sau 1 pentru fiecare eșantion din X_test

# 5. Calculăm matricea de confuzie
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
# - conf_mat e o matrice 2×2:
# [[TN, FP],
# [FN, TP]]
# unde:
# TN = true negatives  (câte exemple 0 au fost ghicite 0)
# FP = false positives (câte exemple 0 au fost ghicite 1)
# FN = false negatives (câte exemple 1 au fost ghicite 0)
# TP = true positives  (câte exemple 1 au fost ghicite 1)

# 6. Calculăm metrici de evaluare
precision = precision_score(y_test, y_pred)
# - precision = TP / (TP + FP), adică dintre predicțiile “1” câte erau cu adevărat “1”
recall = recall_score(y_test, y_pred)
# - recall = TP / (TP + FN), adică dintre exemplele “1” reale câte au fost detectate
accuracy = accuracy_score(y_test, y_pred)
# - accuracy = (TP + TN) / (Total exemple), adică proporția totală de predicții corecte

print(f"\nPrecision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"Accuracy:  {accuracy:.2f}")
