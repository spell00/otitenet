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
from otitenet.data.data_getters import GetData
from otitenet.logging.grad_cam import _resolve_layer, _FeatureHook

def run_debug_heatmap_layer7():
    log_path = "logs/best_models/notNormal/resnet18/otite_ds_64/nsize224/fgsm1/ncal0/softmax_contrastive/no/prototypes_no/npos1/nneg1"
    
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
        path="./data/otite_ds_64",
        valid_dataset='Banque_Viscaino_Chili_2020',
        device='cpu',
        normalize='no',
        new_size=224,
        groupkfold=1, 
        seed=1,
        remove_zeros=False,
        log1p=False
    )
    
    data_getter = GetData(args.path, args.valid_dataset, args)
    data, unique_labels, _ = data_getter.get_variables()
    
    img_arr = data['inputs']['valid'][3] # n6.jpg
    img_chw = img_arr.transpose(2,0,1)
    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    # Target selection
    subcenters = model.get_subcenters()
    pred_idx = 0 # Normal
    sc = subcenters[pred_idx].detach().cpu().numpy()
    grad_cam_proto = np.mean(sc, axis=0)
    reference_embedding = torch.from_numpy(grad_cam_proto).to(device)

    # Compute Heatmap
    target_module = model.model[7] # Layer 7
    hook = _FeatureHook(target_module)
    
    # Forward
    embedding = model(input_tensor)
    if isinstance(embedding, tuple):
        embedding = embedding[0]
    
    # Distance
    distance = F.pairwise_distance(embedding, reference_embedding.unsqueeze(0), p=2)
    score = -distance
    score.backward()
    
    gradients = hook.gradients
    activations = hook.activations
    
    if gradients is None:
        print("Gradients are None!")
    else:
        print(f"Gradients stats: Min={gradients.min():.6f}, Max={gradients.max():.6f}, Mean={gradients.mean():.6f}")
        print(f"Number of zero gradients: {(gradients == 0).sum().item()} / {gradients.numel()}")

    if activations is None:
        print("Activations are None!")
    else:
        print(f"Activations stats: Min={activations.min():.6f}, Max={activations.max():.6f}, Mean={activations.mean():.6f}")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    print(f"Weights stats: Min={weights.min():.6f}, Max={weights.max():.6f}, Mean={weights.mean():.6f}")
    
    weighted_act = (weights * activations).sum(dim=1, keepdim=True)
    print(f"Weighted act stats: Min={weighted_act.min():.6f}, Max={weighted_act.max():.6f}, Mean={weighted_act.mean():.6f}")
    
    cam = torch.relu(weighted_act)
    print(f"CAM after ReLU stats: Min={cam.min():.6f}, Max={cam.max():.6f}, Mean={cam.mean():.6f}")
    
    hook.close()

if __name__ == "__main__":
    run_debug_heatmap_layer7()
