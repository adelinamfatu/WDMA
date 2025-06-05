import pandas as pd

# 1: Creăm un DataFrame cu date de clienți, incluzând duplicate
data = {
    "customer_id": [101, 102, 103, 101, 104, 102],
    "name": ["Alice", "Bob", "Charlie", "Alice", "David", "Bob"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com",
              "alice@example.com", "david@example.com", "bob@example.com"]
}

df = pd.DataFrame(data)

print("Original Customer Database:\n")
print(df)

# 2: Eliminăm duplicatele exacte pe toate coloanele
df_cleaned = df.drop_duplicates()

print("\nDatabase After Removing Exact Duplicates:\n")
print(df_cleaned)

# 3: Eliminăm duplicatele pe baza unei singure coloane (de exemplu, customer_id)
df_cleaned = df.drop_duplicates(subset=["customer_id"], keep="first")

print("\nDatabase After Removing Duplicates Based on Customer ID:\n")
print(df_cleaned)
