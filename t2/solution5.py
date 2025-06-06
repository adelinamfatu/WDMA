import nltk
from nltk.corpus import movie_reviews

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Încărcăm setul de date 'movie_reviews' din NLTK
# - Fiecare recenzie e stocată în corpus ca listă de cuvinte
# - Categoriile posibile sunt 'pos' (pozitive) și 'neg' (negative)
nltk.download('movie_reviews')
documents = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        # Transformăm lista de tokeni în text complet, despărțind prin spații
        review_text = " ".join(movie_reviews.words(fileid))
        documents.append(review_text)
        labels.append(category)

# Convertim etichetele în valori numerice: 'pos' -> 1, 'neg' -> 0
y = [1 if label == 'pos' else 0 for label in labels]

# 2. Împărțim recenziile și etichetele în set de antrenament (70%) și test (30%)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    documents,
    y,
    test_size=0.3,
    random_state=42
)

# 3. Transformăm textul în caracteristici numerice folosind TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english', # Eliminăm cuvintele de legătură frecvente ("and", "the" etc.)
    max_features=3000 # Limităm vocabularul la 3000 de termeni cei mai relevanți
)
# fit_transform învață vocabularul din X_train_texts și transformă textele în matrice TF-IDF
X_train = vectorizer.fit_transform(X_train_texts)
# transform aplică același vocabular pe setul de test
X_test = vectorizer.transform(X_test_texts)

# 4. Antrenăm un model de regresie logistică pe TF-IDF
model = LogisticRegression(max_iter=2000, random_state=42)
# max_iter crește numărul de iterații pentru a asigura convergența
model.fit(X_train, y_train)

# 5. Realizăm predicții pe setul de test
y_pred = model.predict(X_test)

# 6. Evaluăm performanța modelului
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))

# (Opțional) Afișăm câteva predicții aleatorii pentru comparare
random_indices = np.random.choice(len(y_test), size=5, replace=False)
for idx in random_indices:
    print("\nReview:")
    # Afișăm primele 200 de caractere din recenzie
    print(X_test_texts[idx][:200] + "...")
    print(
        f"Predicted Sentiment: {'pos' if y_pred[idx] == 1 else 'neg'} "
        f"| Actual: {'pos' if y_test[idx] == 1 else 'neg'}"
    )
