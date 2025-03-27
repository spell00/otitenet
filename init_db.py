import mysql.connector
from mysql.connector import Error

# Connect to MySQL
try:
    conn = mysql.connector.connect(
        host="localhost",  # Update with your host
        user="y_user",  # Update with your MySQL username
        password="password",  # Update with your MySQL password
        database="results_db"  # Update with your MySQL database name
    )
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) UNIQUE
    )
    ''')

    # Create people table (optional, since you're using it too)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS people (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        name VARCHAR(255),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')

    # Create results table (also used)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        model_name VARCHAR(255),
        task VARCHAR(255),
        path VARCHAR(255),
        nsize VARCHAR(255),
        fgsm VARCHAR(255),
        n_calibration VARCHAR(255),
        classif_loss VARCHAR(255),
        dloss VARCHAR(255),
        prototypes VARCHAR(255),
        npos VARCHAR(255),
        nneg VARCHAR(255),
        pred_label VARCHAR(255),
        confidence FLOAT,
        log_path VARCHAR(255),
        person_id INT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(person_id) REFERENCES people(id)
    )
    ''')

    # Commit changes and close the connection
    conn.commit()

    print("✅ Database initialized.")

except Error as e:
    print(f"❌ Error: {e}")

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
