import mysql.connector
from mysql.connector import Error

# Connect to MySQL
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="y_user",
        password="password",
        database="results_db"
    )
    cursor = conn.cursor()

    # First, check if normalize column exists in results table
    print("Checking results table...")
    cursor.execute("""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'results' AND COLUMN_NAME = 'normalize'
    """)
    if not cursor.fetchone():
        print("Adding normalize column to results table...")
        try:
            cursor.execute("""
                ALTER TABLE results 
                ADD COLUMN normalize VARCHAR(32) DEFAULT 'no'
            """)
            print("✅ Added normalize column to results table")
        except Exception as e:
            print(f"Note: {e}")
    else:
        print("✅ normalize column already exists")

    # Add unique constraint to results table if not already present
    print("\nChecking unique constraint on results table...")
    cursor.execute("""
        SELECT CONSTRAINT_NAME FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME = 'results' AND CONSTRAINT_NAME = 'unique_analysis'
    """)
    if not cursor.fetchone():
        print("Adding unique constraint to results table...")
        try:
            cursor.execute("""
                ALTER TABLE results 
                ADD UNIQUE KEY unique_analysis (filename(100), model_name(50), nsize(20), fgsm(20), normalize(10), 
                                                n_calibration(20), classif_loss(20), dloss(20))
            """)
            print("✅ Added unique_analysis constraint to results table")
        except Exception as e:
            if "Duplicate entry" in str(e):
                print("⚠️  Table may have duplicate entries. Consider checking for duplicates.")
            else:
                print(f"Note: {e}")
    else:
        print("✅ unique_analysis constraint already exists")

    # Create model_usage_summary table with shorter key
    print("\nCreating model_usage_summary table...")
    cursor.execute("""
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
    """)
    print("✅ model_usage_summary table created/verified")

    # Populate model_usage_summary from existing results
    print("\nPopulating model_usage_summary from existing results...")
    cursor.execute("""
        INSERT INTO model_usage_summary 
        (model_name, task, path, nsize, fgsm, normalize, n_calibration, classif_loss, dloss, prototypes, npos, nneg, n_neighbors, num_samples_analyzed)
        SELECT 
            model_name, task, path, nsize, fgsm, COALESCE(normalize, 'no'), n_calibration, classif_loss, dloss, prototypes, npos, nneg, n_neighbors,
            COUNT(*) as num_samples_analyzed
        FROM results
        GROUP BY model_name, task, path, nsize, fgsm, COALESCE(normalize, 'no'), n_calibration, classif_loss, dloss, prototypes, npos, nneg, n_neighbors
        ON DUPLICATE KEY UPDATE 
            num_samples_analyzed = VALUES(num_samples_analyzed)
    """)
    print(f"✅ Populated model_usage_summary ({cursor.rowcount} rows affected)")

    conn.commit()
    print("\n✅ Migration successful: normalize column added and model_usage_summary table created")

except Error as e:
    print(f"❌ Error: {e}")

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
