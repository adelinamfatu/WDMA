import pandas as pd

# 1: Creăm un DataFrame simplu cu o coloană "Gender"
data = {
    "PersonID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Sophia", "Noah"],
    "Gender": ["Male", "Female", "Male", "Female", "Male"]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# 2: Convertim valorile din coloana "Gender" în 0/1
# - Folosim metoda .map() care ia un dicționar de conversie
# - "Male" devine 0, "Female" devine 1
df["Gender_Binary"] = df["Gender"].map({"Male": 0, "Female": 1})

print("\nDataset After Encoding Gender as Binary:\n")
print(df)
