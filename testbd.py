import duckdb

conn = duckdb.connect("tasks.db")

result = conn.execute("SELECT id, text, topic, difficulty, keywords, latex_example FROM tasks LIMIT 10").fetchall()
for row in result:
    print(row)

conn.close()