import sqlite3


def create_candles_table(db, table_name: str):
    query = f"""
    CREATE TABLE IF NOT EXISTS  {table_name}(
        ticker TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        PRIMARY KEY (ticker, timestamp)
    )
    """
    db.execute_query(query)
