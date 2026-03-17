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

def run_debug_layer7():
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
        new_size=224,
        fgsm='1',
        n_calibration='0',
        classif_loss='softmax_contrastive',
        dloss='dloss',
        prototypes_to_use='no',
        n_positives='1',
        n_negatives='1',
        normalize='no',
        dist_fct='euclidean',
        groupkfold=1, 
        seed=1,
        path="./data/otite_ds_64",
        valid_dataset='Banque_Viscaino_Chili_2020',
        device='cpu',
        remove_zeros=False,
        log1p=False
    )
    
    data_getter = GetData(args.path, args.valid_dataset, args)
    data, _, _ = data_getter.get_variables()
    
    img_arr = data['inputs']['valid'][3] # n6.jpg
    img_chw = img_arr.transpose(2,0,1)
    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    # Hook layer 7 (layer4 in resnet)
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    
    handle = model.model[7].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    handle.remove()
    
    act = activations[0]
    print(f"Layer 7 (layer4) activations shape: {act.shape}")
    print(f"Mean activation: {act.mean().item()}")
    print(f"Max activation: {act.max().item()}")
    print(f"Number of zeros: {(act == 0).sum().item()} / {act.numel()}")

if __name__ == "__main__":
    run_debug_layer7()
