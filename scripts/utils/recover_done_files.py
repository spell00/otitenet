import mysql.connector
import os
import itertools

# Connect to database
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="y_user",
        password="password",
        database="results_db"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM best_models_registry")
    rows = cursor.fetchall()
except Exception as e:
    print(f"Error reading DB: {e}")
    rows = []

test_tag = "PROD"
task = "notNormal"
logs_dir = "logs"

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

count = 0
for row in rows:
    # row keys: model_name, nsize, fgsm, prototypes, npos, nneg, dloss, dist_fct, classif_loss, n_calibration, normalize
    model = row.get("model_name")
    fgsm = row.get("fgsm")
    n_calibration = row.get("n_calibration")
    dloss = row.get("dloss")
    normalize = row.get("normalize")
    classif_loss = row.get("classif_loss")
    prototype = row.get("prototypes", "no")
    nneg = row.get("nneg", 1)

    exp_id = None
    
    # 1. Siamese arcface/triplet with varied prototypes
    if classif_loss in ["arcface", "triplet"] and prototype in ["batch", "no", "class"]:
        if classif_loss == "triplet":
            # the n_negatives loop creates:
            # exp_id="${test_tag}_${task}_siamese_${model}_triplet_${dloss}_nneg${n_negatives}_proto_no_ncal${n_calibration}_norm${normalize}_fgsm${fgsm}"
            if nneg in [1, 5] and prototype == "no":
                exp_id = f"{test_tag}_{task}_siamese_{model}_triplet_{dloss}_nneg{nneg}_proto_no_ncal{n_calibration}_norm{normalize}_fgsm{fgsm}"
        if not exp_id:
            exp_id = f"{test_tag}_{task}_siamese_{model}_{classif_loss}_{dloss}_proto{prototype}_ncal{n_calibration}_norm{normalize}_fgsm{fgsm}"

    # 2. Siamese ce/hinge
    elif classif_loss in ["ce", "hinge"]:
        exp_id = f"{test_tag}_{task}_siamese_{model}_{classif_loss}_{dloss}_proto_no_ncal{n_calibration}_norm{normalize}_fgsm{fgsm}"
    
    # 3. Siamese softmax_contrastive
    elif classif_loss == "softmax_contrastive":
        exp_id = f"{test_tag}_{task}_siamese_{model}_softmax_contrastive_{dloss}_proto_no_ncal{n_calibration}_norm{normalize}_fgsm{fgsm}"
    
    if exp_id:
        filepath = os.path.join(logs_dir, f"{exp_id}.done")
        with open(filepath, "w") as f:
            f.write("Recovered from DB\n")
        count += 1
        
print(f"Recovered {count} completed siamese runs from results_db and created .done files.")

# For cnn_mlp runs, we will check existing logs and if they don't have errors at the end, mark them done.
# But train_cnn_mlp_compare.py doesn't write DB, so we look at logs/cnn_mlp_*.log files directly to recover them.
# Given they were named cnn_mlp_${job_id}_${model}_${variant}_${classif_loss}_${normalize}.log
import glob
cnn_mlp_logs = glob.glob("logs/cnn_mlp_*.log")
cnn_mlp_count = 0
for cl in cnn_mlp_logs:
    if "_error.log" in cl: continue
    # See if file ended normally. If yes, extract params to create ${exp_id}.done
    # Example format: cnn_mlp_12_resnet18_cnn_transfer_hinge_yes.log
    # But we need fgsm, dloss, n_calibration to recreate exp_id exactly.
    # We might not have them natively in the old filename.
    pass

print(f"To also skip completed CNN/MLP logs, ensure they have valid '.done' files if needed.")
