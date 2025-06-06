import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Generăm un set de date sintetic cu clasificare, având multe caracteristici (features)
X, y = make_classification(
    n_samples=200, # numărul total de eșantioane (rânduri)
    n_features=10, # numărul total de caracteristici pe care le generăm
    n_informative=5, # câte caracteristici să fie cu adevărat informative pentru țintă
    n_redundant=2, # câte caracteristici să fie redundante (liniare combinații ale celor informative)
    random_state=42 # seed pentru reproducibilitatea setului generat
)

# (Opțional) Transformăm în DataFrame pandas pentru a vizualiza mai ușor
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
# df arată astfel:
#    feature_0  feature_1  ...  feature_9  target
# 0   -0.302152   1.356240  ...   0.811457       0
# 1    0.404764  -0.751357  ...   1.128222       1
# ...

# 2. Împărțim datele în set de antrenament și test (70% antrenament, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 30% din eșantioane merg în setul de test
    random_state=42 # seed pentru a păstra aceeași împărțire de fiecare dată
)

# 3. Antrenăm o regresie logistică cu regularizare L1 (penalizare L1)
# - penalty='l1' indică folosirea regularizării L1 (care forțează coeficienții să devină exact 0)
# - solver='saga' suportă L1 pentru clasificare binară și este eficient pentru seturi mari
# - max_iter=1000 crește numărul de iterații pentru a asigura convergența
model_l1 = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=1000,
    random_state=42
)
model_l1.fit(X_train, y_train)
# - După fit, coeficienții modelului (model_l1.coef_) pot avea multe valori exact 0 datorită penalizării L1

# 4. Antrenăm o regresie logistică cu regularizare L2 (penalizare L2)
model_l2 = LogisticRegression(
    penalty='l2',
    solver='saga',
    max_iter=1000,
    random_state=42
)
model_l2.fit(X_train, y_train)
# - În acest caz, coeficienții nu vor fi forțați la zero; vor fi în continuare nenuli, dar micșorați

# 5. Comparăm coeficienții celor două modele
coeff_l1 = model_l1.coef_[0] # vectorul de coeficienți pentru clasa pozitivă (binarii)
coeff_l2 = model_l2.coef_[0] # la fel pentru L2

# Construim un DataFrame cu comparația: fiecare caracteristică și cei doi coeficienți
coef_comparison = pd.DataFrame({
    "Feature": feature_names,
    "L1_Coefficient": coeff_l1,
    "L2_Coefficient": coeff_l2
})
print("Coefficient Comparison (L1 vs L2):")
print(coef_comparison)
# - Coloanele L1_Coefficient pot conține multe 0 (semnificând că acele feature-uri au fost eliminate de regularizare)
# - Coloanele L2_Coefficient vor avea valori mai mici decât în mod normal, dar rareori exact 0

# 6. Evaluăm acuratețea pe setul de test
acc_l1 = model_l1.score(X_test, y_test)
acc_l2 = model_l2.score(X_test, y_test)
print(f"\nTest Accuracy - L1: {acc_l1:.2f}")
print(f"Test Accuracy - L2: {acc_l2:.2f}")
# - score(X_test, y_test) echivalează cu accuracy_score(y_test, model.predict(X_test))
# - Comparăm performanța modelelor cu penalizări diferite

# 7. Observăm care caracteristici rămân semnificative sub L1
# - Un coeficient nenul sub L1 indică un feature păstrat de model ca relevant
non_zero_features = coef_comparison[
    coef_comparison["L1_Coefficient"] != 0]["Feature"].tolist()
print("\nFeatures with non-zero coefficients under L1:")
print(non_zero_features)
# - Afișează lista celor feature-uri care nu au fost complet eliminate de regularizarea L1
