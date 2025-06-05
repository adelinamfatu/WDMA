import pandas as pd

# 1: Încărcăm setul de date MovieLens 100K direct de pe URL
# - URL-ul u.data conține ratings în format tab-delimited
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
column_names = ["user_id", "movie_id", "rating", "timestamp"]
# - Folosim sep="\t" deoarece fișierul e separat prin taburi
# - names=column_names definește numele coloanelor
# - usecols selectează doar coloanele relevante: user_id, movie_id, rating
ratings = pd.read_csv(
    url,
    sep="\t",
    names=column_names,
    usecols=["user_id", "movie_id", "rating"]
)

# - URL-ul u.item conține informații despre filme (movie_id și title printre altele)
url_movies = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
# - Fișierul e separat prin pipe "|" și are encoding latin-1
# - names=["movie_id", "title"] și usecols=[0,1] pastram doar primele două coloane
movies = pd.read_csv(
    url_movies,
    sep="|",
    encoding="latin-1",
    names=["movie_id", "title"],
    usecols=[0, 1]
)

# 2: Combinăm (merge) titlurile filmelor cu dataset-ul de ratings pe baza movie_id
# - ratings.merge(movies, on="movie_id") adaugă coloana "title" în DataFrame-ul ratings
ratings = ratings.merge(movies, on="movie_id")

# 3: Filtrăm utilizatorii care au acordat mai puțin de 10 ratinguri
# - groupby("user_id").size() returnează un Series cu numărul de ratinguri per user
user_ratings_count = ratings.groupby("user_id").size()
# - valid_users = index-ul celor care au >= 10 ratinguri
valid_users = user_ratings_count[user_ratings_count >= 10].index
# - folosim isin(valid_users) pentru a păstra doar rândurile cu useri valizi
filtered_ratings = ratings[ratings["user_id"].isin(valid_users)]

# 4: Calculăm ratingul mediu per film
# - groupby("title") grupează după titlul filmului
# - ["rating"].mean() calculează media ratingurilor pentru fiecare titlu
movie_avg_ratings = filtered_ratings.groupby("title")["rating"].mean()

# 5: Obținem top 5 filme cele mai bine cotate (media cea mai mare)
# - sort_values(ascending=False) sortează descrescător după valoarea medie a ratingului
top_movies = movie_avg_ratings.sort_values(ascending=False).head(5)

# 6: Afișăm lista celor mai recomandate 5 filme
print("Top 5 Most Popular Movies:\n")
print(top_movies)
