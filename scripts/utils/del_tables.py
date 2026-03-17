import mysql.connector
from mysql.connector import Error


PRESERVED_TABLES = {"best_models_registry"}


def clear_app_computed_tables():
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db",
        )
        cursor = conn.cursor()

        cursor.execute("SHOW TABLES")
        all_tables = [row[0] for row in cursor.fetchall()]
        tables_to_clear = [table for table in all_tables if table not in PRESERVED_TABLES]

        if not tables_to_clear:
            print("ℹ️ No app-computed tables found to clear.")
            return

        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        for table_name in tables_to_clear:
            cursor.execute(f"TRUNCATE TABLE `{table_name}`")
            print(f"🧹 Cleared table: {table_name}")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

        conn.commit()
        print("✅ Cleared all app-computed tables. Preserved: best_models_registry")

    except Error as e:
        print(f"❌ Error: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    clear_app_computed_tables()
