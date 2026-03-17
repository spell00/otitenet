"""
OPTIMIZED migration script to add model_rank column to best_models_registry table
Uses bulk updates for better performance with large datasets
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
    
    print("Checking if model_rank column exists...")
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
            ADD COLUMN model_rank INT DEFAULT NULL,
            ADD INDEX idx_model_rank (model_rank)
        """)
        conn.commit()
        print("✅ Column added successfully")
    else:
        print("ℹ️  model_rank column already exists")
    
    # Now populate/update the model_rank using a more efficient approach
    print("Updating model ranks based on MCC (this may take a moment)...")
    
    # Use a single UPDATE with a subquery for better performance
    cursor.execute("""
        UPDATE best_models_registry bmr
        INNER JOIN (
            SELECT 
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY 
                        CONCAT_WS('|',
                            COALESCE(model_name, ''),
                            COALESCE(nsize, ''),
                            COALESCE(fgsm, ''),
                            COALESCE(prototypes, ''),
                            COALESCE(npos, ''),
                            COALESCE(nneg, ''),
                            COALESCE(dloss, ''),
                            COALESCE(dist_fct, ''),
                            COALESCE(classif_loss, ''),
                            COALESCE(n_calibration, ''),
                            COALESCE(normalize, ''),
                            COALESCE(n_neighbors, '')
                        )
                    ORDER BY mcc DESC
                ) as config_rank,
                ROW_NUMBER() OVER (ORDER BY mcc DESC) as global_rank
            FROM best_models_registry
        ) ranked ON bmr.id = ranked.id
        SET bmr.model_rank = CASE 
            WHEN ranked.config_rank = 1 THEN ranked.global_rank
            ELSE NULL 
        END
    """)
    
    rows_affected = cursor.rowcount
    conn.commit()
    
    print(f"✅ Migration successful: Updated {rows_affected} rows")
    
    # Show summary
    cursor.execute("""
        SELECT COUNT(*) 
        FROM best_models_registry
        WHERE model_rank IS NOT NULL
    """)
    
    num_ranked = cursor.fetchone()[0]
    print(f"📊 {num_ranked} unique model configurations ranked")
    
    cursor.execute("""
        SELECT model_rank, model_name, mcc, dist_fct, normalize
        FROM best_models_registry
        WHERE model_rank IS NOT NULL
        ORDER BY model_rank
        LIMIT 10
    """)
    
    print("\nTop 10 ranked models:")
    print("Rank | Model Name      | MCC   | Dist Fct | Normalize")
    print("-" * 65)
    for row in cursor.fetchall():
        print(f"#{row[0]:3d} | {row[1]:15s} | {row[2]:.3f} | {str(row[3]):8s} | {str(row[4])}")

except Error as e:
    print(f"❌ Error: {e}")
    if conn:
        conn.rollback()

finally:
    if conn and conn.is_connected():
        cursor.close()
        conn.close()
        print("\n✅ Migration complete!")
