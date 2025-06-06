import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Încărcăm setul de date Wine (3 clase de vin)
wine = load_wine()
X = wine.data # Matricea de caracteristici (tabel cu 13 coloane)
y = wine.target # Vectorul țintă (valorile 0, 1 sau 2)

# (Opțional) Dacă vrem să vedem primele rânduri într-un DataFrame:
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())

# 2. Împărțim datele în seturi de antrenament (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% dintre exemple merg la test
    random_state=42 # seed pentru reproducibilitate
)

# 3. Antrenăm un clasificator Naïve Bayes gaussian (algorithmul din Exercise 1)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
# - GaussianNB calculează media și varianța fiecărei caracteristici pentru fiecare clasă
# - y_pred_nb va conține etichetele prezise (0, 1, 2)
y_pred_nb = nb_model.predict(X_test)

# 4. Antrenăm un model de regresie logistică
# - max_iter=2000: crește numărul de iterații pentru a evita avertismente de neconvergență
logreg_model = LogisticRegression(max_iter=2000)
logreg_model.fit(X_train, y_train)
# - y_pred_logreg va conține etichetele prezise (0, 1, 2)
y_pred_logreg = logreg_model.predict(X_test)

# 5. Comparăm metricile: acuratețe, precizie și recall pentru fiecare model
# - average='macro' folosește media aritmetică a metricilor pe cele 3 clase,
# tratând fiecare clasă cu aceeași importanță
metrics = {}
for model_name, y_pred in [("Naive Bayes", y_pred_nb), ("Logistic Regression", y_pred_logreg)]:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

# 6. Afișăm rezultatele pentru fiecare model
for model_name, scores in metrics.items():
    print(f"=== {model_name} ===")
    print(f"Accuracy:  {scores['Accuracy']:.2f}")
    print(f"Precision: {scores['Precision']:.2f}")
    print(f"Recall:    {scores['Recall']:.2f}")
    print()

# Opțional: Matricea de confuzie pentru fiecare model (pentru analiză detaliată)
# from sklearn.metrics import confusion_matrix
# print("Naive Bayes Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_nb))
# print("\nLogistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_logreg))
