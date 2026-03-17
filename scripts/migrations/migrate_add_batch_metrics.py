#!/usr/bin/env python3
"""
Migration script to add batch effect metrics columns to best_models_registry table
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

    print("Adding batch effect metric columns to best_models_registry...")

    # Check and add batch_entropy_norm column
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'results_db' 
        AND TABLE_NAME = 'best_models_registry' 
        AND COLUMN_NAME = 'batch_entropy_norm'
    """)
    
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            ALTER TABLE best_models_registry 
            ADD COLUMN batch_entropy_norm FLOAT NULL AFTER mcc
        """)
        print("✅ Added batch_entropy_norm column")
    else:
        print("ℹ️  batch_entropy_norm column already exists")
    
    # Check and add batch_nmi column
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'results_db' 
        AND TABLE_NAME = 'best_models_registry' 
        AND COLUMN_NAME = 'batch_nmi'
    """)
    
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            ALTER TABLE best_models_registry 
            ADD COLUMN batch_nmi FLOAT NULL AFTER batch_entropy_norm
        """)
        print("✅ Added batch_nmi column")
    else:
        print("ℹ️  batch_nmi column already exists")
    
    # Check and add batch_ari column
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'results_db' 
        AND TABLE_NAME = 'best_models_registry' 
        AND COLUMN_NAME = 'batch_ari'
    """)
    
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            ALTER TABLE best_models_registry 
            ADD COLUMN batch_ari FLOAT NULL AFTER batch_nmi
        """)
        print("✅ Added batch_ari column")
    else:
        print("ℹ️  batch_ari column already exists")

    conn.commit()
    print("\n✅ Migration successful: batch effect columns added to best_models_registry")

except Error as e:
    print(f"❌ Migration error: {e}")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals() and conn.is_connected():
        conn.close()
