import os
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mysql.connector
from PIL import Image
from torchvision import transforms
from otitenet.models.cnn import Net
from otitenet.data.data_getters import GetData, PerImageNormalize
from otitenet.logging.grad_cam import log_grad_cam_similarity, log_grad_cam_all_classes

def extract_params_from_log_path(log_path: str):
    """Derive parameters from the stored best-model log path."""
    params = {}
    if not log_path:
        return params
    parts = log_path.strip("/").split("/")
    try:
        base_idx = parts.index("best_models")
    except ValueError:
        return params

    if len(parts) > base_idx + 1:
        params["task"] = parts[base_idx + 1]
    if len(parts) > base_idx + 2:
        params["model_name"] = parts[base_idx + 2]
    if len(parts) > base_idx + 3:
        params["dataset_name"] = parts[base_idx + 3]
    return params

def get_all_models(limit=None):
    conn = mysql.connector.connect(
        host="localhost",
        user="y_user",
        password="password",
        database="results_db"
    )
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM best_models_registry ORDER BY mcc DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

def build_model_descriptor(row: dict, params_ext: dict) -> str:
    # Compose a stable, descriptive filename for the model + params
    parts = []
    parts.append(str(row.get('model_name', 'unknown')))
    if params_ext.get('dataset_name'):
        parts.append(str(params_ext.get('dataset_name')))
    # Common hyperparams
    def _fmt(key, label=None):
        val = row.get(key)
        if val is None or str(val) == "":
            return None
        return f"{label or key}-{val}"
    for key, label in [
        ('nsize', 'nsize'),
        ('fgsm', 'fgsm'),
        ('n_calibration', 'ncal'),
        ('classif_loss', 'loss'),
        ('dloss', 'dloss'),
        ('prototypes', 'prot'),
        ('npos', 'npos'),
        ('nneg', 'nneg'),
        ('dist_fct', 'dist'),
        ('normalize', 'norm'),
        ('n_neighbors', 'k')
    ]:
        s = _fmt(key, label)
        if s:
            parts.append(s)
    return "-".join(parts)

def get_prediction_from_prototypes(embedding, class_prototypes, dist_fct_name='euclidean'):
    best_dist = float('inf')
    best_label = None
    
    emb_tensor = torch.as_tensor(embedding)
    if emb_tensor.ndim == 1:
        emb_tensor = emb_tensor.unsqueeze(0)
    
    if dist_fct_name == 'cosine':
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

    for label, proto in class_prototypes.items():
        if proto is None: continue
        proto_tensor = torch.as_tensor(proto)
        if proto_tensor.ndim == 1:
            proto_tensor = proto_tensor.unsqueeze(0)
            
        if dist_fct_name == 'cosine':
            proto_tensor = F.normalize(proto_tensor, p=2, dim=1)
            sim = torch.mm(emb_tensor, proto_tensor.t())
            dist = 1.0 - torch.max(sim).item()
        else:
            dists = torch.cdist(emb_tensor, proto_tensor)
            dist = torch.min(dists).item()
            
        if dist < best_dist:
            best_dist = dist
            best_label = label
            
    return best_label

def run_batch(model_limit=None, sample_limit=10):
    models = get_all_models(limit=model_limit)
    if not models:
        print("No models found.")
        return

    # Batch settings
    layers = [3, 4, 5, 6, 7]
    alpha = 0.55
    target_valid_dataset = 'Banque_Viscaino_Chili_2020'

    for m_idx, model_row in enumerate(models, start=1):
        log_path = model_row['log_path']
        params_ext = extract_params_from_log_path(log_path)
        
        model_path = os.path.join(log_path, 'model.pth')
        if not os.path.exists(model_path):
            print(f"Skipping: model file not found for {model_row.get('model_name')} at {model_path}")
            continue
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Checkpoint dims
        n_batches_checkpoint = state_dict['dann.weight'].shape[0]
        n_cats_checkpoint = state_dict['linear.weight'].shape[0]
        n_subcenters_checkpoint = state_dict['subcenters'].shape[1]
        
        print(f"[{m_idx}/{len(models)}] Model: {model_row['model_name']} Task: {params_ext.get('task')} Dataset: {params_ext.get('dataset_name')}")
        print(f"Checkpoint dims: n_cats={n_cats_checkpoint}, n_batches={n_batches_checkpoint}, n_subcenters={n_subcenters_checkpoint}")
        
        # Init prediction log for this model
        pred_log_rows = []

        args = argparse.Namespace(
            model_name=model_row['model_name'],
            task=params_ext.get('task', 'otitis'),
            new_size=model_row.get('nsize', 224),
            fgsm=model_row.get('fgsm', '0'),
            n_calibration=model_row.get('n_calibration', '0'),
            classif_loss=model_row.get('classif_loss', 'ce'),
            dloss=model_row.get('dloss', 'dloss'),
            prototypes_to_use=model_row.get('prototypes', 'all'),
            n_positives=str(model_row.get('npos', '1')),
            n_negatives=str(model_row.get('nneg', '1')),
            normalize=model_row.get('normalize', 'no'),
            dist_fct=model_row.get('dist_fct', 'euclidean'),
            groupkfold=1, 
            seed=1,
            path=f"./data/{params_ext.get('dataset_name','')}",
            valid_dataset=target_valid_dataset,
            device='cpu',
            remove_zeros=False,
            log1p=False
        )

        # Load Data
        data_getter = GetData(args.path, args.valid_dataset, args)
        data, unique_labels, unique_batches = data_getter.get_variables()

        # Load Model
        model = Net(args.device, n_cats=n_cats_checkpoint, n_batches=n_batches_checkpoint,
                    model_name=args.model_name, is_stn=0,
                    n_subcenters=n_subcenters_checkpoint)
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()

        # Load Prototypes
        proto_path = os.path.join(log_path, 'prototypes.pkl')
        if not os.path.exists(proto_path):
            print(f"Skipping: prototypes not found for {model_row.get('model_name')} at {proto_path}")
            continue
        with open(proto_path, 'rb') as f:
            proto_obj = pickle.load(f)
        class_protos = proto_obj.class_prototypes['train']

        # Output Directories (old + new)
        old_batch_out_dir = os.path.join(log_path, f"batch_grad_cam_{args.valid_dataset}")
        os.makedirs(old_batch_out_dir, exist_ok=True)
        
        model_desc = build_model_descriptor(model_row, params_ext)

        # Process Images (class-balanced sampling)
        valid_inputs = data['inputs']['valid']
        valid_names = data['names']['valid']
        valid_labels = np.array(data['labels']['valid'])

        # Build per-class index lists
        per_class_indices = {}
        for lbl in unique_labels:
            try:
                idxs = np.where(valid_labels == lbl)[0]
            except Exception:
                # Fallback for mixed types: compare as strings
                lbl_str = str(lbl)
                idxs = np.array([i for i, v in enumerate(valid_labels) if str(v) == lbl_str])
            per_class_indices[lbl] = idxs.tolist()

        # Determine selected indices per class
        selected_indices = []
        if sample_limit is None or sample_limit <= 0:
            # Use all validation samples
            selected_indices = list(range(len(valid_inputs)))
            total_selected = len(selected_indices)
            print(f"Processing all {total_selected} images for {model_desc}...")
        else:
            # Cap per class
            rng = np.random.default_rng(1)
            for lbl, idxs in per_class_indices.items():
                if not idxs:
                    continue
                # Shuffle deterministically for reproducibility
                idxs_arr = np.array(idxs)
                if len(idxs_arr) > 1:
                    rng.shuffle(idxs_arr)
                take_n = min(int(sample_limit), len(idxs_arr))
                selected_indices.extend(idxs_arr[:take_n].tolist())
            total_selected = len(selected_indices)
            print(f"Processing {total_selected} images (per-class limit {int(sample_limit)}) for {model_desc}...")

        for i_idx, i in enumerate(selected_indices, start=1):
            raw_arr = valid_inputs[i]
            name = valid_names[i]
            base_name = os.path.splitext(name)[0]
            
            # Robust CHW conversion
            if raw_arr.ndim == 2: # (H, W)
                 img_chw = np.stack([raw_arr] * 3, axis=0)
            elif raw_arr.ndim == 3:
                 if raw_arr.shape[0] == 1: # (1, H, W)
                      img_chw = np.concatenate([raw_arr] * 3, axis=0)
                 elif raw_arr.shape[2] == 1: # (H, W, 1)
                      img_chw = np.concatenate([raw_arr] * 3, axis=2).transpose(2, 0, 1)
                 elif raw_arr.shape[0] == 3: # (3, H, W)
                      img_chw = raw_arr
                 elif raw_arr.shape[2] == 3: # (H, W, 3)
                      img_chw = raw_arr.transpose(2, 0, 1)
                 else:
                      img_chw = np.stack([raw_arr[:,:,0]] * 3, axis=0)
            else:
                 img_chw = np.zeros((3, 224, 224), dtype=np.float32)

            input_tensor = torch.from_numpy(img_chw).float().unsqueeze(0).to(args.device)
            inputs_dict = {'queries': {'inputs': [input_tensor.cpu()]}}
            
            # Compute prediction for old-path single-class Grad-CAM
            with torch.no_grad():
                outputs = model(input_tensor)
                embedding = outputs[0] if isinstance(outputs, tuple) else outputs
                embedding = embedding.detach().cpu().numpy()
            pred_label = get_prediction_from_prototypes(embedding, class_protos, dist_fct_name=args.dist_fct)
            grad_cam_proto = None
            if pred_label in list(proto_obj.class_prototypes['train'].keys()):
                 grad_cam_proto = proto_obj.class_prototypes['train'][pred_label]
            elif args.classif_loss != 'ce':
                 try:
                     if hasattr(model, 'get_subcenters'):
                         subcenters = model.get_subcenters()
                         # Map label to index
                         try:
                             pred_idx = list(unique_labels).index(pred_label)
                         except Exception:
                             pred_idx = 0
                         sc = subcenters[pred_idx].detach().cpu().numpy()
                         grad_cam_proto = np.mean(sc, axis=0) if sc.ndim == 2 else sc
                 except Exception as e:
                     print(f"Error getting subcenters for {name}: {e}")

            if (i_idx % 10) == 0 or i_idx == 1:
                print(f"[{i_idx}/{total_selected}] {name} -> {pred_label}")

            for layer in layers:
                # Detect existing outputs to avoid recomputation
                layer_dir_old = os.path.join(old_batch_out_dir, f"layer{layer}")
                overlay_old_path = None
                if grad_cam_proto is not None:
                    overlay_old_path = os.path.join(layer_dir_old, f"{base_name}_{pred_label}_grad_cam.png")
                old_already_done = overlay_old_path is not None and os.path.exists(overlay_old_path)

                xai_dir = os.path.join('logs', 'xAI', 'grad-cam', args.valid_dataset, base_name, f"layer{layer}")
                pred_tag = pred_label if pred_label is not None else "unknown"
                xai_filename = f"{model_desc}__pred-{pred_tag}"
                expected_xai = [
                    os.path.join(xai_dir, f"{xai_filename}_class{label}.png")
                    for label, proto in class_protos.items() if proto is not None
                ]
                # If there are no usable prototypes, consider the xAI layer done
                xai_already_done = len(expected_xai) == 0 or all(os.path.exists(p) for p in expected_xai)

                if old_already_done and xai_already_done:
                    continue

                if not old_already_done and grad_cam_proto is not None:
                    os.makedirs(layer_dir_old, exist_ok=True)
                    out_filename_old = f"{base_name}_{pred_label}"
                    log_grad_cam_similarity(
                        model, 0, inputs_dict, 'queries', layer_dir_old, out_filename_old,
                        grad_cam_proto, device=args.device, layer=layer, alpha=alpha
                    )
                
                if not xai_already_done:
                    os.makedirs(xai_dir, exist_ok=True)
                    log_grad_cam_all_classes(
                        model, 0, inputs_dict, 'queries', xai_dir, xai_filename,
                        class_protos, device=args.device, layer=layer, alpha=alpha
                    )
    
    print("Batch processing complete. See logs/xAI/grad-cam/<dataset>/<image>/layer*/ for consolidated outputs across models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Grad-CAM generator across all models (best to worst by MCC).")
    parser.add_argument("--model-limit", type=int, default=None, help="Limit number of models to process (best-first by MCC).")
    parser.add_argument("--sample-limit", type=int, default=10, help="Max samples per class to process per model (default: 10). Use 0 or negative to process all samples.")
    args_cli = parser.parse_args()

    sample_limit = args_cli.sample_limit
    if sample_limit is not None and sample_limit <= 0:
        sample_limit = None

    run_batch(model_limit=args_cli.model_limit, sample_limit=sample_limit)
