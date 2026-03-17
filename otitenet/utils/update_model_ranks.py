"""
Helper function to update model ranks in best_models_registry
This should be called after any INSERT/UPDATE to the registry
"""
import mysql.connector
from mysql.connector import Error

def update_model_ranks():
    """
    Recompute and update model_rank for all models in best_models_registry.
    This ensures consistent ranking even when new models are added.
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="y_user",
            password="password",
            database="results_db"
        )
        cursor = conn.cursor()
        
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
            placeholders = ','.join(['%s'] * len(unranked_ids))
            cursor.execute(f"""
                UPDATE best_models_registry 
                SET model_rank = NULL 
                WHERE id IN ({placeholders})
            """, tuple(unranked_ids))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return len(unique_models)
        
    except Error as e:
        print(f"❌ Error updating model ranks: {e}")
        if conn:
            conn.rollback()
        return 0

if __name__ == "__main__":
    # Can be run standalone to update ranks
    num_ranked = update_model_ranks()
    print(f"✅ Updated ranks for {num_ranked} models")
