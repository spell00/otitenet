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

def run_check_range():
    log_path = "logs/best_models/notNormal/resnet18/otite_ds_64/nsize224/fgsm1/ncal0/softmax_contrastive/no/prototypes_no/npos1/nneg1"
    
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
    
    valid_inputs = data['inputs']['valid']
    print(f"Data type: {valid_inputs.dtype}")
    print(f"Min: {valid_inputs.min()}, Max: {valid_inputs.max()}")
    print(f"Shape of one image: {valid_inputs[0].shape}")

if __name__ == "__main__":
    run_check_range()
