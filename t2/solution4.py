import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Încărcăm setul de date Spambase dintr-un fișier CSV
# - Fără header, deoarece fișierul nu conține etichete de coloană
df = pd.read_csv("spambase.csv", header=None)

# 2. Separăm caracteristicile (X) și eticheta țintă (y)
# - Primele 57 coloane sunt feature-uri numerice (frecvențe de cuvinte, caractere etc.)
# - Ultima coloană (58) este eticheta: 1 = spam, 0 = not spam
X = df.iloc[:, :-1] # toate coloanele, mai puțin ultima
y = df.iloc[:, -1] # doar ultima coloană

# 3. Împărțim în seturi de antrenament (70%) și test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 30% la test
    random_state=42 # seed pentru reproducibilitate
)

# 4. Antrenăm un clasificator Naïve Bayes (MultinomialNB e potrivit pentru date de frecvențe)
model = MultinomialNB()
model.fit(X_train, y_train)
# - fit() calculează probabilitățile condiționate ale fiecărei caracteristici date etichetei (spam/ham)

# 5. Facem predicții pe setul de test
y_pred = model.predict(X_test)
# - predict() returnează 0 sau 1 pentru fiecare exemplu din X_test

# 6. Evaluăm modelul
accuracy = accuracy_score(y_test, y_pred)
# - accuracy_score = (număr exemple prezise corect) / (număr total exemple)

conf_mat = confusion_matrix(y_test, y_pred)
# - confusion_matrix returnează matricea:
# [[TN, FP],
# [FN, TP]]
# TN = număr ham prezise corect, FP = ham prezise greșit spam
# FN = spam prezis greșit ham,      TP = spam prezis corect

print(f"Test Accuracy: {accuracy:.3f}")
print("\nConfusion Matrix:")
print(conf_mat)
