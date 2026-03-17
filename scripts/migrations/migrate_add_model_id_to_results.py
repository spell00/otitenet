"""
Migration to add `model_id` to results and backfill from best_models_registry.
"""
import mysql.connector
from mysql.connector import Error

DB_NAME = "results_db"
FK_NAME = "fk_results_model_id"


def column_exists(cursor):
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = 'results'
          AND COLUMN_NAME = 'model_id'
        """,
        (DB_NAME,),
    )
    return cursor.fetchone()[0] > 0


def fk_exists(cursor):
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS
        WHERE CONSTRAINT_SCHEMA = %s
          AND CONSTRAINT_NAME = %s
        """,
        (DB_NAME, FK_NAME),
    )
    return cursor.fetchone()[0] > 0


try:
    conn = mysql.connector.connect(
        host="localhost",
        user="y_user",
        password="password",
        database=DB_NAME,
    )
    cursor = conn.cursor()

    added = False
    if not column_exists(cursor):
        cursor.execute("ALTER TABLE results ADD COLUMN model_id INT NULL AFTER person_id")
        added = True
        print("✅ Added model_id column to results")
    else:
        print("ℹ️  model_id column already exists on results")

    if not fk_exists(cursor):
        cursor.execute(
            f"""
            ALTER TABLE results
            ADD CONSTRAINT {FK_NAME}
            FOREIGN KEY (model_id) REFERENCES best_models_registry(id)
            ON DELETE SET NULL
            """
        )
        print("✅ Added foreign key from results.model_id to best_models_registry.id")
    else:
        print("ℹ️  Foreign key already present for results.model_id")

    # Backfill using log_path match when available
    cursor.execute(
        """
        UPDATE results r
        JOIN best_models_registry b ON r.log_path = b.log_path
        SET r.model_id = b.id
        WHERE r.model_id IS NULL
        """
    )
    updated = cursor.rowcount
    conn.commit()
    print(f"✅ Backfilled model_id for {updated} rows using log_path matches")

except Error as e:
    print(f"❌ Migration failed: {e}")
    if conn:
        conn.rollback()
finally:
    if conn and conn.is_connected():
        cursor.close()
        conn.close()
