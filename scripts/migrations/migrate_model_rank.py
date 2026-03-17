"""
Migration script to add model_rank column to best_models_registry table
and populate it based on MCC ranking
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
    
    # Check if model_rank column exists
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'results_db' 
        AND TABLE_NAME = 'best_models_registry' 
        AND COLUMN_NAME = 'model_rank'
    """)
    
    if cursor.fetchone()[0] == 0:
        print("Adding model_rank column to best_models_registry...")
        cursor.execute("""
            ALTER TABLE best_models_registry 
            ADD COLUMN model_rank INT DEFAULT NULL
        """)
        conn.commit()
        print("✅ Column added successfully")
    else:
        print("ℹ️  model_rank column already exists")
    
    # Now populate/update the model_rank based on current MCC ranking
    print("Updating model ranks based on MCC...")
    
    # Get all models ordered by MCC descending
    cursor.execute("""
        SELECT id, model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, 
               classif_loss, n_calibration, normalize, n_neighbors, mcc
        FROM best_models_registry
        ORDER BY mcc DESC
    """)
    
    all_models = cursor.fetchall()
    
    # Build dedupe key for each model (same logic as in app.py)
    def make_dedupe_key(row):
        return "|".join([
            str(row[1] or ""),  # model_name
            str(row[2] or ""),  # nsize
            str(row[3] or ""),  # fgsm
            str(row[4] or ""),  # prototypes
            str(row[5] or ""),  # npos
            str(row[6] or ""),  # nneg
            str(row[7] or ""),  # dloss
            str(row[8] or ""),  # dist_fct
            str(row[9] or ""),  # classif_loss
            str(row[10] or ""), # n_calibration
            str(row[11] or ""), # normalize
            str(row[12] or ""), # n_neighbors
        ])
    
    # Deduplicate keeping only the best (highest MCC) for each configuration
    seen_keys = {}
    unique_models = []
    
    for row in all_models:
        key = make_dedupe_key(row)
        if key not in seen_keys:
            seen_keys[key] = True
            unique_models.append(row)
    
    # Assign ranks (1-indexed)
    print(f"Assigning ranks to {len(unique_models)} unique model configurations...")
    
    for rank, row in enumerate(unique_models, start=1):
        model_id = row[0]
        cursor.execute("""
            UPDATE best_models_registry 
            SET model_rank = %s 
            WHERE id = %s
        """, (rank, model_id))
    
    # Set rank to NULL for duplicate configurations (not the best one)
    all_ids = {row[0] for row in all_models}
    ranked_ids = {row[0] for row in unique_models}
    unranked_ids = all_ids - ranked_ids
    
    if unranked_ids:
        print(f"Setting rank to NULL for {len(unranked_ids)} duplicate configurations...")
        placeholders = ','.join(['%s'] * len(unranked_ids))
        cursor.execute(f"""
            UPDATE best_models_registry 
            SET model_rank = NULL 
            WHERE id IN ({placeholders})
        """, tuple(unranked_ids))
    
    conn.commit()
    print(f"✅ Migration successful: model_rank column added and {len(unique_models)} models ranked")
    
    # Show summary
    cursor.execute("""
        SELECT model_rank, model_name, mcc, dist_fct, normalize
        FROM best_models_registry
        WHERE model_rank IS NOT NULL
        ORDER BY model_rank
        LIMIT 10
    """)
    
    print("\nTop 10 ranked models:")
    print("Rank | Model Name | MCC | Dist Fct | Normalize")
    print("-" * 60)
    for row in cursor.fetchall():
        print(f"#{row[0]:3d} | {row[1]:15s} | {row[2]:.3f} | {row[3]:8s} | {row[4]}")

except Error as e:
    print(f"❌ Error: {e}")
    if conn:
        conn.rollback()

finally:
    if conn and conn.is_connected():
        cursor.close()
        conn.close()
