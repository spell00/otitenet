import mysql.connector
from mysql.connector import Error

# Connect to MySQL
try:
    conn = mysql.connector.connect(
        host="localhost",  # Update with your host
        user="y_user",     # Update with your MySQL username
        password="password",  # Update with your MySQL password
        database="results_db"  # Update with your MySQL database name
    )
    cursor = conn.cursor()

    # Disable foreign key checks
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

    # Get all table names
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    for (table_name,) in tables:
        cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
        print(f"🗑️ Dropped table: {table_name}")

    # Re-enable foreign key checks
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    conn.commit()
    print("✅ All tables deleted.")

except Error as e:
    print(f"❌ Error: {e}")

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
