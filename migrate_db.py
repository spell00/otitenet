#!/usr/bin/env python3
"""
Migration script to add normalize column to best_models_registry table
"""
import mysql.connector
from mysql.connector import Error

try:
    # Connect to the database
    conn = mysql.connector.connect(
        host="localhost",
        user="y_user",
        password="password",
        database="results_db"
    )
    cursor = conn.cursor()

    # Check if normalize column exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'results_db' 
        AND TABLE_NAME = 'best_models_registry' 
        AND COLUMN_NAME = 'normalize'
    """)
    
    if cursor.fetchone()[0] == 0:
        # Add normalize column
        cursor.execute("""
            ALTER TABLE best_models_registry 
            ADD COLUMN normalize VARCHAR(32) DEFAULT 'no'
        """)
        print("✅ Added normalize column")
    else:
        print("ℹ️  normalize column already exists")
    
    # Drop old unique key and create new one with normalize
    cursor.execute("SHOW INDEX FROM best_models_registry WHERE Key_name = 'unique_combo'")
    has_index = cursor.fetchone()
    cursor.fetchall()  # Consume remaining results
    
    if has_index:
        cursor.execute("""
            ALTER TABLE best_models_registry
            DROP INDEX unique_combo
        """)
        print("✅ Dropped old unique_combo index")
    
    cursor.execute("""
        ALTER TABLE best_models_registry
        ADD UNIQUE KEY unique_combo (model_name, fgsm, prototypes, npos, nneg, dloss, n_calibration, normalize)
    """)

    conn.commit()
    print("✅ Migration successful: normalize column added to best_models_registry")

except Error as e:
    print(f"❌ Migration error: {e}")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals() and conn.is_connected():
        conn.close()
