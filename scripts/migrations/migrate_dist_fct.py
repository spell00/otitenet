"""
Migration script to add dist_fct column to results and model_usage_summary tables.
This ensures consistent model matching using all relevant parameters.
"""
import mysql.connector
from mysql.connector import Error

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="y_user",
        password="password",
        database="results_db"
    )
    cursor = conn.cursor()

    print("Checking results table for dist_fct...")
    cursor.execute("SHOW COLUMNS FROM results LIKE 'dist_fct'")
    if not cursor.fetchone():
        print("Adding dist_fct to results table...")
        cursor.execute("ALTER TABLE results ADD COLUMN dist_fct VARCHAR(32) AFTER dloss")
        conn.commit()
        print("✅ Added dist_fct to results")
    else:
        print("ℹ️ dist_fct already exists in results")

    print("Checking model_usage_summary table for dist_fct...")
    cursor.execute("SHOW COLUMNS FROM model_usage_summary LIKE 'dist_fct'")
    if not cursor.fetchone():
        print("Adding dist_fct to model_usage_summary table...")
        cursor.execute("ALTER TABLE model_usage_summary ADD COLUMN dist_fct VARCHAR(32) AFTER dloss")
        conn.commit()
        print("✅ Added dist_fct to model_usage_summary")
    else:
        print("ℹ️ dist_fct already exists in model_usage_summary")

    # Update unique keys to include dist_fct
    print("Updating unique keys...")
    
    # results table
    try:
        cursor.execute("ALTER TABLE results DROP INDEX unique_analysis")
        cursor.execute("""
            ALTER TABLE results 
            ADD UNIQUE KEY unique_analysis (filename(100), model_name(50), nsize(20), fgsm(20), normalize(10), n_calibration(20), classif_loss(20), dloss(20), dist_fct(20))
        """)
        conn.commit()
        print("✅ Updated unique key for results")
    except Exception as e:
        print(f"⚠️ Could not update unique key for results: {e}")

    # model_usage_summary table
    try:
        cursor.execute("ALTER TABLE model_usage_summary DROP INDEX unique_model_config")
        cursor.execute("""
            ALTER TABLE model_usage_summary 
            ADD UNIQUE KEY unique_model_config (model_name, task, nsize, fgsm, normalize, n_calibration, classif_loss, dloss, dist_fct, prototypes, npos, nneg)
        """)
        conn.commit()
        print("✅ Updated unique key for model_usage_summary")
    except Exception as e:
        print(f"⚠️ Could not update unique key for model_usage_summary: {e}")

except Error as e:
    print(f"❌ Error during migration: {e}")
    if conn:
        conn.rollback()
finally:
    if conn and conn.is_connected():
        cursor.close()
        conn.close()
        print("Migration complete.")
