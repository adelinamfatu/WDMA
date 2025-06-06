import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 1. Generăm un set de date sintetic cu dezechilibru puternic (90% non-boală, 10% boală)
# - n_samples=1000: 1000 de eșantioane
# - n_features=5: 5 caracteristici numerice
# - n_informative=3: 3 dintre ele vor ajuta la prezicerea țintei
# - weights=[0.90, 0.10]: clasa 0 (healthy) va fi 90%, clasa 1 (disease) 10%
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    weights=[0.90, 0.10],
    random_state=42
)

# 2. Împărțim datele în set de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 3. Antrenăm regresie logistică cu class_weight='balanced'
# - class_weight='balanced' ajustează automat ponderile claselor invers proporțional cu frecvența lor
# - astfel modelul acordă mai multă atenție clasei minoritare (disease)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. Facem predicții pe setul de test
y_pred = model.predict(X_test)

# 5. Evaluăm performanța cu matricea de confuzie și raportul de clasificare
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# - Afișează [[TN, FP], [FN, TP]] pentru clasa 0 (healthy) și clasa 1 (disease)

print("\nClassification Report (precision, recall, F1-score):")
print(classification_report(y_test, y_pred))
# - Afișează precision, recall, f1-score și support pentru fiecare clasă

# 6. (Opțional) Comparăm cu model fără class_weight
model_no_weight = LogisticRegression(random_state=42)
model_no_weight.fit(X_train, y_train)
y_pred_no_weight = model_no_weight.predict(X_test)

print("------- Without Class Weighting -------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_no_weight))
print("\nClassification Report (precision, recall, F1-score):")
print(classification_report(y_test, y_pred_no_weight))
