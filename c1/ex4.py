import sqlite3

# 1: Conectare la baza de date SQLite
# Dacă fisierul "transactions.db" nu există, va fi creat automat
conn = sqlite3.connect("transactions.db")
cursor = conn.cursor()

# 2: Creare tabel “transactions” (dacă nu există deja)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        amount REAL,
        transaction_type TEXT,
        date TEXT
    )
''')

# 3: Inserare date de exemplu (numai dacă tabelul e gol)
# 1) Verificăm câte rânduri există în tabel
cursor.execute("SELECT COUNT(*) FROM transactions")
if cursor.fetchone()[0] == 0:
    # 2) Definim un set de date sample (tupluri cu user_id, amount, type, date)
    sample_data = [
        (1, 250.75, 'deposit', '2024-02-23'),
        (1, -100.50, 'withdrawal', '2024-02-24'),
        (2, 500.00, 'deposit', '2024-02-20'),
        (3, -75.00, 'withdrawal', '2024-02-21'),
    ]
    # 3) Inserăm toate rândurile cu executemany (folosind parametrizare “?,”)
    cursor.executemany("INSERT INTO transactions (user_id, amount, transaction_type, date) VALUES (?, ?, ?, ?)", sample_data)
    # 4) Salvăm modificările în baza de date
    conn.commit()

# 4: Interogare și extragere tranzacții pentru un anumit utilizator
user_id = 1
# Executăm SELECT cu parametrizare pentru a evita SQL injection
cursor.execute("SELECT * FROM transactions WHERE user_id = ?", (user_id,))
transactions = cursor.fetchall() # preluăm toate rândurile rezultate ca listă de tuple

# 5: Afișăm rezultatele
print(f"Transactions for User {user_id}:\n")
for txn in transactions:
    # txn e o tuple: (id, user_id, amount, transaction_type, date)
    print(f"ID: {txn[0]}, Amount: {txn[2]}, Type: {txn[3]}, Date: {txn[4]}")

# Închidem cursorul și conexiunea la baza de date
cursor.close()
conn.close()
