import os

import mysql.connector
from mysql.connector import Error


def db_config() -> dict:
    return {
        "host": os.environ.get("OTITENET_DB_HOST", "localhost"),
        "user": os.environ.get("OTITENET_DB_USER", "y_user"),
        "password": os.environ.get("OTITENET_DB_PASSWORD", "password"),
        "database": os.environ.get("OTITENET_DB_NAME", "results_db"),
    }


# Connect to MySQL
conn = None
cursor = None
try:
    conn = mysql.connector.connect(**db_config())
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
        normalize VARCHAR(255),
        n_calibration VARCHAR(255),
        classif_loss VARCHAR(255),
        dloss VARCHAR(255),
        prototypes VARCHAR(255),
        npos VARCHAR(255),
        nneg VARCHAR(255),
        n_neighbors VARCHAR(255),
        pred_label VARCHAR(255),
        confidence FLOAT,
        log_path VARCHAR(255),
        person_id INT,
        model_id INT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY unique_analysis (filename(100), model_name(50), nsize(20), fgsm(20), normalize(10), n_calibration(20), classif_loss(20), dloss(20)),
        FOREIGN KEY(person_id) REFERENCES people(id),
        FOREIGN KEY(model_id) REFERENCES best_models_registry(id) ON DELETE SET NULL
    )
    ''')

    # Create best_models_registry table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS best_models_registry (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(64),
        nsize VARCHAR(32),
        fgsm VARCHAR(32),
        prototypes VARCHAR(64),
        npos INT,
        nneg INT,
        dloss VARCHAR(64),
        dist_fct VARCHAR(32),
        classif_loss VARCHAR(32),
        n_calibration VARCHAR(32),
        accuracy FLOAT,
        mcc FLOAT,
        batch_entropy_norm FLOAT,
        batch_nmi FLOAT,
        batch_ari FLOAT,
        normalize VARCHAR(32),
        n_neighbors int DEFAULT 1,
        prototype_strategy VARCHAR(32),
        prototype_components INT,
        log_path VARCHAR(255),
        model_rank INT,
        UNIQUE KEY unique_combo (model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss, n_calibration, normalize, n_neighbors, prototype_strategy, prototype_components),
        INDEX idx_model_rank (model_rank)
    )
    ''')

    # Create model_usage_summary table to track which models have been used for analysis
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_usage_summary (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(64),
        task VARCHAR(64),
        path VARCHAR(255),
        nsize VARCHAR(32),
        fgsm VARCHAR(32),
        normalize VARCHAR(32),
        n_calibration VARCHAR(32),
        classif_loss VARCHAR(32),
        dloss VARCHAR(32),
        prototypes VARCHAR(32),
        npos VARCHAR(32),
        nneg VARCHAR(32),
        n_neighbors VARCHAR(32),
        num_samples_analyzed INT DEFAULT 0,
        last_used DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY unique_model_config (model_name, task, nsize, fgsm, normalize, n_calibration, classif_loss, dloss, prototypes, npos, nneg)
    )
    ''')
    # Commit changes and close the connection
    conn.commit()

    print("✅ Database initialized.")

except Error as e:
    print(f"❌ Error: {e}")

finally:
    if conn and conn.is_connected():
        cursor.close()
        conn.close()
