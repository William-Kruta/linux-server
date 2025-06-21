import sqlite3


class DatabaseConnection:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn = None
        self.cursor = None

    def __enter__(self):
        print(f"[DB] Connecting to {self.db_name}")
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        return self  # Enables usage like: with DatabaseConnection(...) as db:

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("[DB] Connection closed.")

    def execute(self, query: str, params: tuple = ()):
        print(f"[DB] Executing: {query}")
        self.cursor.execute(query, params)
        self.conn.commit()

    def executemany(self, query: str, params: tuple = ()):
        print(f"[DB] Executing: {query}")
        self.cursor.executemany(query, params)
        self.conn.commit()

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def list_tables(self):
        """
        Lists all tables in the SQLite database.

        Returns:
            A list of table names.
        """
        try:
            self.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.fetchall()
            return [table[0] for table in tables]  # Extract table names from tuples
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []

    def get_table_columns(self, table_name: str):
        try:

            # Execute a PRAGMA table_info() query to get column information
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = self.cursor.fetchall()
            # Extract the column names from the results
            column_names = [column[1] for column in columns_info]
            return column_names

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return []

    def drop_table(self, table_name: str):
        """
        Drops a table from the SQLite database.

        Args:
            table_name: The name of the table to drop.
        """
        try:
            self.execute(f"DROP TABLE IF EXISTS {table_name}")
            print(f"Table '{table_name}' dropped successfully.")
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
