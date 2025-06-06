import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Creăm un dataset mic cu două caracteristici (age, income) și o etichetă binară (purchased)
data = {
    'age': [25, 40, 35, 50, 28, 60, 45],
    'income': [50000, 70000, 60000, 80000, 52000, 100000, 75000],
    'purchased': [0, 1, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
#    - df arată așa:
#       age  income  purchased
#    0   25   50000          0
#    1   40   70000          1
#    2   35   60000          0
#    3   50   80000          1
#    4   28   52000          0
#    5   60  100000          1
#    6   45   75000          1

# 2. Separăm caracteristicile (X) și eticheta țintă (y)
X = df[['age', 'income']]  # DataFrame cu coloanele folosite ca features
y = df['purchased'] # Serie cu eticheta “purchased” (0 = nu a cumpărat, 1 = a cumpărat)

# 3. Împărțim datele în seturi de antrenament și test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% dintre rânduri merg în setul de test
    random_state=42 # număr fix pentru reproducibilitate
)
# - După split, X_train și y_train conțin 80% din date (5 rânduri), iar X_test și y_test 20% (2 rânduri)

# 4. Inițializăm și antrenăm modelul de regresie logistică
model = LogisticRegression()
model.fit(X_train, y_train)
# - model.fit antrenează regresia logistică pe X_train (age + income) și y_train

# 5. Realizăm predicții pe setul de test
y_pred = model.predict(X_test)
# - model.predict returnează etichete (0 sau 1) prezise pentru fiecare rând din X_test

# 6. Evaluăm performanța folosind acuratețea simplă
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.2f}")
# - (y_pred == y_test) creează un array boolean cu True acolo unde predicția e corectă
# - .mean() calculează proporția de True (acuratețea)
# - Afișăm acuratețea cu două zecimale, de exemplu “0.50” pentru 50%

# Opțional: afișăm dataset-ul inițial pentru verificare
# print("\nDataset:")
# print(df)
