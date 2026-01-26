import sqlite3
import pandas as pd

# connect to the sqlite database
conn = sqlite3.connect('mydb.db')
# create cursor object to execute sql queries
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

table_names = cursor.fetchall()

for table in table_names:
    print(table[0])

# execute a select query to retrieve all records from a table
cursor.execute("SELECT * FROM metrics")

# fetch all the rows
rows = cursor.fetchall()

# print column names
column_names = [desc[0] for desc in cursor.description]

# preview table
pd.DataFrame(rows,columns=column_names)

# Execute a SELECT query to retrieve all records from a table
cursor.execute('SELECT * FROM params')

# Fetch all the rows
rows = cursor.fetchall()

# Print column names
column_names = [desc[0] for desc in cursor.description]

# preview table
pd.DataFrame(rows, columns=column_names)