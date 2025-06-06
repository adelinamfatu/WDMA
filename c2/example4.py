import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Creăm un dataset mic cu trei caracteristici și eticheta “loan_approved”
# Fiecare tuplu reprezintă: (income, credit_score, debt_ratio, loan_approved)
data = [
    (50000, 700, 0.30, 1),
    (30000, 600, 0.40, 0),
    (80000, 750, 0.20, 1),
    (40000, 580, 0.50, 0),
    (75000, 720, 0.35, 1),
    (28000, 550, 0.45, 0),
    (90000, 780, 0.15, 1),
    (32000, 600, 0.42, 0),
    (66000, 710, 0.38, 1),
    (25000, 530, 0.50, 0)
]
df = pd.DataFrame(
    data,
    columns=["income", "credit_score", "debt_ratio", "loan_approved"]
)
# - “income”: venitul anual
# - “credit_score”: scorul de credit
# - “debt_ratio”: raportul datoriilor (parte din venit)
# - “loan_approved”: 1 dacă împrumutul a fost aprobat, 0 dacă nu

# 2. Separăm caracteristicile (X) și ținta (y)
X = df[["income", "credit_score", "debt_ratio"]]
y = df["loan_approved"]

# 3. Împărțim datele în set de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% din date merg la test
    random_state=42 # seed pentru reproducibilitate
)

# 4. Antrenăm un clasificator de tip Decision Tree
model = DecisionTreeClassifier(
    max_depth=3, # limităm adâncimea arborelui pentru claritate
    random_state=42 # seed pentru stabilitatea rezultatelor
)
model.fit(X_train, y_train)
# - fit învață structura arborelui pe baza caracteristicilor și etichetelor din X_train, y_train

# 5. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - predict returnează 0 sau 1 pentru fiecare rând din X_test

# 6. Evaluăm performanța folosind acuratețea simplă
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")
# - (y_pred == y_test) creează un array boolean cu True pentru p
