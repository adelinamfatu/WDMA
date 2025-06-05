import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1: Creăm un DataFrame cu recenzii de produse
reviews = [
    "This phone has a great camera and amazing battery life.",
    "The laptop performance is very fast and smooth.",
    "Terrible customer service, very disappointing experience.",
    "The product quality is top-notch and highly recommended.",
    "The delivery was late and the packaging was damaged."
]

df = pd.DataFrame({"Review": reviews})

print("Original Product Reviews:\n")
print(df)

# 2: Aplicăm TF-IDF Vectorization pentru a extrage cuvintele-cheie
# - stop_words="english": elimină cuvinte comune în engleză (de ex. "and", "the")
# - max_features=5: extrage doar primele 5 cuvinte-cheie după scor TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5)
tfidf_matrix = vectorizer.fit_transform(df["Review"])
# - fit_transform(): construiește vocabularul pe baza tuturor recenziilor și returnează matricea TF-IDF (5 coloane, câte un cuvânt-cheie)

# 3: Obținem numele caracteristicilor (cuvintele-cheie extrase)
keywords = vectorizer.get_feature_names_out()

print("\nExtracted Keywords from Product Reviews:\n")
print(keywords)
