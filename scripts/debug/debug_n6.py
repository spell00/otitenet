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
from otitenet.logging.grad_cam import log_grad_cam_similarity

def extract_params_from_log_path(log_path: str):
    params = {}
    parts = log_path.strip("/").split("/")
    try:
        base_idx = parts.index("best_models")
    except ValueError:
        return params
    if len(parts) > base_idx + 1: params["task"] = parts[base_idx + 1]
    if len(parts) > base_idx + 2: params["model_name"] = parts[base_idx + 2]
    if len(parts) > base_idx + 3: params["dataset_name"] = parts[base_idx + 3]
    return params

def run_debug_n6():
    log_path = "logs/best_models/notNormal/resnet18/otite_ds_64/nsize224/fgsm1/ncal0/softmax_contrastive/no/prototypes_no/npos1/nneg1"
    params_ext = extract_params_from_log_path(log_path)
    
    # Load model
    model_path = os.path.join(log_path, 'model.pth')
    state_dict = torch.load(model_path, map_location='cpu')
    n_batches = state_dict['dann.weight'].shape[0]
    n_cats = state_dict['linear.weight'].shape[0]
    n_subcenters = state_dict['subcenters'].shape[1]
    
    device = 'cpu'
    model = Net(device, n_cats=n_cats, n_batches=n_batches,
                model_name='resnet18', is_stn=0,
                n_subcenters=n_subcenters)
    model.load_state_dict(state_dict)
    model.eval()

    # Load data
    args = argparse.Namespace(
        model_name='resnet18',
        task='notNormal',
        new_size=224,
        fgsm='1',
        n_calibration='0',
        classif_loss='softmax_contrastive',
        dloss='dloss',
        prototypes_to_use='no',
        n_positives='1',
        n_negatives='1',
        normalize='no',
        dist_fct='euclidean', # or 'cosine'? User path says prototypes_no, but dist_fct?
        groupkfold=1, 
        seed=1,
        path=f"./data/{params_ext['dataset_name']}",
        valid_dataset='Banque_Viscaino_Chili_2020',
        device='cpu',
        remove_zeros=False,
        log1p=False
    )
    
    data_getter = GetData(args.path, args.valid_dataset, args)
    data, unique_labels, _ = data_getter.get_variables()
    
    valid_inputs = data['inputs']['valid']
    valid_names = data['names']['valid']
    
    n6_idx = -1
    for i, name in enumerate(valid_names):
        if name == 'n6.jpg':
            n6_idx = i
            break
    
    if n6_idx == -1:
        print("n6.jpg not found in valid set")
        return

    img_arr = valid_inputs[n6_idx]
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
         img_chw = img_arr.transpose(2, 0, 1)
    else:
         img_chw = img_arr

    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        embedding = outputs[0] if isinstance(outputs, tuple) else outputs
        embedding_np = embedding.detach().cpu().numpy()

    # Get Predicted Label via Prototypes (as in my script)
    with open(os.path.join(log_path, 'prototypes.pkl'), 'rb') as f:
        proto_obj = pickle.load(f)
    class_protos = proto_obj.class_prototypes['train']
    
    from run_batch_grad_cam import get_prediction_from_prototypes
    # We need to know the dist_fct used during training.
    # User path shows 'softmax_contrastive'.
    pred_label = get_prediction_from_prototypes(embedding_np, class_protos, dist_fct_name='euclidean')
    
    print(f"n6.jpg Predicted: {pred_label}")
    print(f"Unique labels: {unique_labels}")
    
    # Try getting subcenter
    subcenters = model.get_subcenters()
    pred_idx = np.where(unique_labels == pred_label)[0][0]
    sc = subcenters[pred_idx].detach().cpu().numpy()
    sc_mean = np.mean(sc, axis=0) if sc.ndim == 2 else sc
    
    print(f"Subcenter mean (first 5): {sc_mean[:5]}")
    
    # Try getting training prototype
    tp = class_protos[pred_label]
    tp_mean = np.mean(tp, axis=0) if tp.ndim == 2 else tp
    print(f"Training Proto mean (first 5): {tp_mean[:5]}")
    
    # Cosine similarity between sc_mean and tp_mean
    cos_sim = np.dot(sc_mean, tp_mean) / (np.linalg.norm(sc_mean) * np.linalg.norm(tp_mean))
    print(f"Cosine Similarity (Subcenter vs Training Proto): {cos_sim:.4f}")

    # Generate Grad-CAM for n6 to see if it errors
    out_dir = "debug_grad_cam"
    os.makedirs(out_dir, exist_ok=True)
    inputs_dict = {'queries': {'inputs': [torch.from_numpy(img_chw).unsqueeze(0)]}}
    
    print("Computing Grad-CAM for n6 (layer 7)...")
    log_grad_cam_similarity(
        model, 0, inputs_dict, 'queries', out_dir, 'n6_debug',
        sc_mean, device=device, layer=7, alpha=0.55
    )
    print(f"Saved to {out_dir}/n6_debug_grad_cam_layer7.png")

if __name__ == "__main__":
    run_debug_n6()
