import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Generăm un set sintetic dezechilibrat (99% tranzacții legitime, 1% frauduloase)
# - n_samples=2000: 2000 de exemple
# - n_features=6: 6 caracteristici pentru fiecare tranzacție
# - weights=[0.99, 0.01]: 99% aparține clasei “legit” (0), 1% aparține clasei “fraudă” (1)
X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.99, 0.01],
    random_state=42
)

# (Opțional) Trebuie convertit într-un DataFrame pentru claritate, cu nume
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["is_fraudulent"] = y
# - df arată coloanele feature_0 ... feature_5 și o coloană binary “is_fraudulent”

# 2. Împărțim datele în set de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 30% pentru test
    random_state=42 # seed fix pentru reproducibilitate
)

# 3. Antrenăm o regresie logistică cu ponderi pentru clasa minoritară
# - class_weight='balanced' ajustează automat ponderile invers proporțional cu frecvența clasei
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
# - Astfel, modelul va acorda mai multă atenție detectării tranzacțiilor frauduloase (evenimente rare)

# 4. Realizăm predicții pe setul de test
y_pred = model.predict(X_test)

# 5. Evaluăm performanța cu matricea de confuzie și raportul de clasificare
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
# - [[TN, FP],
#   [FN, TP]]
# TN = tranzacții legitime ghicite corect
# FP = tranzacții legitime clasificate greșit ca frauduloase
# FN = tranzacții frauduloase clasificate greșit ca legitime
# TP = tranzacții frauduloase detectate corect

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# - Afișează pentru fiecare clasă (0 = legitim, 1 = fraudă):
# * precision: din tranzacțiile semnalate ca frauduloase, câte erau cu adevărat fraude
# * recall: din toate tranzacțiile frauduloase reale, câte au fost detectate
# * f1-score: media armonică între precision și recall
# * support: numărul total de exemple din fiecare clasă

# 6. (Opțional) Afișăm coeficienții modelului pentru a vedea ce caracteristici indică fraudă
coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
})
print("\nFeature Coefficients (Logistic Regression):")
print(coefficients)
# - Valorile pozitive mari indică acele caracteristici a căror creștere crește probabilitatea de fraudă.
# - Valorile negative mari indică acele caracteristici a căror creștere scade probabilitatea de fraudă.
