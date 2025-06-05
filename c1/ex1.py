##  1: Extracția și transformarea datelor
## Extracția și transformarea datelor

import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- Definirea setului de date ---
# Avem o lista mică de emailuri; unele sunt spam (1), altele nu (ham, 0)
emails = [
    "Win a free lottery now",             # probabil spam
    "Meeting at 10 AM",                    # probabil ham
    "You have won a cash prize",           # spam
    "Let's schedule a call",               # ham
    "Exclusive offer just for you",        # spam
    "Please review the attached document", # ham
    "Congratulations, you are selected!",  # spam
    "Reminder: Project deadline tomorrow", # ham
    "Claim your free gift today",          # spam
    "Discounts available only for today",  # spam
    "Schedule your doctor appointment",    # ham
    "Invoice for your recent transaction", # ham
    "Limited time sale on electronics",    # spam
    "Join our webinar this weekend",       # ham
    "Urgent: Update your account information",  # spam
    "Dinner plans for Friday night"        # ham
]

# Etichete corecte pentru emailuri: 1 = Spam, 0 = Ham
labels = np.array([
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 0, 0, 1, 0, 1, 0
])

# Introducerea etichetelor incorecte
np.random.seed(42)

# Calculăm câte etichete să inversăm (50% din total)
num_flips = int(len(labels) * 0.5)

# Alegem aleator 50% dintre indexuri pentru a le "flip-ui"
flip_indices = np.random.choice(len(labels), num_flips, replace=False)

# Copiem etichetele originale și le inversam la indicii selectați
incorrect_labels = labels.copy()
incorrect_labels[flip_indices] = 1 - incorrect_labels[flip_indices]
# Acum incorrect_labels conține etichete greșite pentru 50% dintre exemple

# Transformarea textului în caracteristici numerice
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
# X este o matrice sparse în care fiecare rând reprezintă un email și
# fiecare coloană reprezintă contorul unui cuvânt din vocabularul întregii liste

# Împărțirea datelor în seturi de antrenament și test
# Pentru etichete corecte:
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)
# X_train, y_train: 80% date de antrenament (corecte)
# X_test, y_test: 20% date de test (corecte)

# Pentru etichete incorecte: folosim aceeași proporție, dar pe incorrect_labels
X_train_wrong, _, y_train_wrong, _ = train_test_split(
    X, incorrect_labels, test_size=0.2, random_state=42
)
# Folosim doar X_train_wrong și y_train_wrong pentru antrenarea modelului cu etichete greșite

# --- Antrenarea modelului Naïve Bayes ---
# Model antrenat pe date cu etichete corecte
model_correct = MultinomialNB()
model_correct.fit(X_train, y_train) # ajustăm modelul pe X_train și y_train (corect)

# Model antrenat pe date cu etichete greșite
model_wrong = MultinomialNB()
model_wrong.fit(X_train_wrong, y_train_wrong)  # ajustăm modelul pe X_train_wrong și y_train_wrong (incorect)

# --- Realizarea predicțiilor pe setul de test (care are etichete corecte) ---
y_pred_correct = model_correct.predict(X_test) # predicții cu modelul antrenat pe etichete corecte
y_pred_wrong = model_wrong.predict(X_test) # predicții cu modelul antrenat pe etichete greșite

# --- Evaluarea performanței ---
# Afișăm rapoartele de clasificare (precizie, recall, f1-score) pentru fiecare model
print("=== Model Trained on Correct Labels ===")
print(classification_report(y_test, y_pred_correct)) # comparăm predicțiile cu adevăratele y_test

print("\n=== Model Trained on Incorrect Labels ===")
print(classification_report(y_test, y_pred_wrong)) # comparăm predicțiile cu y_test (care sunt corecte)

# Calculăm și afișăm acuratețea globală pentru ambele modele
accuracy_correct = accuracy_score(y_test, y_pred_correct)
accuracy_wrong = accuracy_score(y_test, y_pred_wrong)
print(f"\nAccuracy with Correct Labels: {accuracy_correct:.2f}") # ex: 0.75
print(f"Accuracy with Incorrect Labels: {accuracy_wrong:.2f}") # probabil mai mic decât modelul corect
