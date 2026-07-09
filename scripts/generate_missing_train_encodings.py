"""Generate train_encodings.npz for models that are missing it."""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, 'src')

from otitenet.app.artifact_registry import load_best_models_registry
from otitenet.app.model_loading import load_model_and_prototypes
from otitenet.data.data_getters import get_images_loaders
from otitenet.utils.utils import get_empty_traces
from otitenet.train.train_triplet_new import TrainAE


def generate_train_encodings(model_dir, output_dir=None):
    """Generate train_encodings.npz for a given model directory.
    
    Args:
        model_dir: Path to the model directory containing model.pth and prototypes.pkl
        output_dir: Path to save train_encodings.npz (defaults to model_dir)
    """
    if output_dir is None:
        output_dir = model_dir
    
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    
    # Check if model files exist
    if not (model_dir / "model.pth").exists():
        print(f"❌ model.pth not found in {model_dir}")
        return False
    if not (model_dir / "prototypes.pkl").exists():
        print(f"❌ prototypes.pkl not found in {model_dir}")
        return False
    
    # Load run metadata to get model configuration
    metadata_path = model_dir / "run_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
    else:
        print(f"⚠️ No run_metadata.json found, will use defaults")
        metadata = {}
    
    # Create minimal args object
    class Args:
        pass
    
    _args = Args()
    
    # Set basic args from metadata or defaults
    _args.task = metadata.get('task', 'notNormal')
    _args.model_name = metadata.get('model_name', 'resnet18')
    _args.path = str(model_dir)
    _args.new_size = 224
    _args.bs = 32
    _args.n_neighbors = 0
    _args.dloss = 'no'
    _args.normalize = 'no' if 'normno' in str(model_dir) else 'yes'
    _args.random_recs = 0
    _args.valid_dataset = metadata.get('valid_dataset', '')
    _args.groupkfold = False
    _args.prototypes_to_use = 'no' if 'prototypes_no' in str(model_dir) else 'class'
    
    # Load data
    try:
        data_path = metadata.get('data_path', 'data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference')
        if not os.path.exists(data_path):
            print(f"⚠️ Data path {data_path} not found, trying alternatives...")
            # Try to find the data directory
            for possible_path in [
                'data/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference',
                'data/otite_ds_64/USA_Turquie_Chili_GMFUNL',
                'data/datasets/otite_ds_64/USA_Turquie_Chili_GMFUNL_inference',
            ]:
                if os.path.exists(possible_path):
                    data_path = possible_path
                    print(f"✅ Found data at {data_path}")
                    break
        
        if not os.path.exists(data_path):
            print(f"❌ Could not find data directory")
            return False
        
        # Load training data
        from otitenet.data.data_getters import get_data
        data = get_data(data_path)
        
        # Get unique labels and batches
        unique_labels = sorted(data['train']['label'].unique())
        unique_batches = sorted(data['train']['batch'].unique()) if 'batch' in data['train'] else []
        
        print(f"Found {len(unique_labels)} unique labels, {len(unique_batches)} unique batches")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load prototypes
    try:
        import pickle
        with open(model_dir / "prototypes.pkl", 'rb') as f:
            prototypes = pickle.load(f)
        print(f"✅ Loaded prototypes")
    except Exception as e:
        print(f"❌ Error loading prototypes: {e}")
        prototypes = {}
    
    # Initialize training wrapper for encoding
    try:
        train = TrainAE(_args, _args.path, load_tb=False, log_metrics=False, keep_models=True,
                        log_inputs=False, log_plots=False, log_tb=False, log_tracking=False,
                        log_mlflow=False, groupkfold=_args.groupkfold)
        train.n_batches = len(unique_batches)
        train.n_cats = len(unique_labels)
        train.unique_batches = unique_batches
        train.unique_labels = unique_labels
        train.epoch = 1
        train.params = {
            'n_neighbors': _args.n_neighbors,
            'lr': 0,
            'wd': 0,
            'smoothing': 0,
            'is_transform': 0,
            'valid_dataset': _args.valid_dataset
        }
        train.set_arcloss()
        print(f"✅ Initialized training wrapper")
    except Exception as e:
        print(f"❌ Error initializing training wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Build data loaders
    try:
        lists, traces = get_empty_traces()
        loaders = get_images_loaders(
            data=data,
            random_recs=_args.random_recs,
            weighted_sampler=0,
            is_transform=0,
            samples_weights=None,
            epoch=1,
            unique_labels=unique_labels,
            triplet_dloss=_args.dloss, bs=_args.bs,
            prototypes_to_use=_args.prototypes_to_use,
            prototypes=prototypes,
            size=_args.new_size,
            normalize=_args.normalize,
            batch_encoder=None
        )
        print(f"✅ Built data loaders")
    except Exception as e:
        print(f"❌ Error building data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load model and encode training data
    try:
        model, _, _, _, _, _, _, _, _ = load_model_and_prototypes(_args)
        train.model = model
        print(f"✅ Loaded model")
        
        with torch.no_grad():
            _, lists, _ = train.loop('train', None, 0, loaders['train'], lists, traces)
        
        train_encs = np.concatenate(lists['train']['encoded_values'])
        train_cats = np.concatenate(lists['train']['cats'])
        
        print(f"✅ Encoded {len(train_encs)} training samples")
        
    except Exception as e:
        print(f"❌ Error encoding training data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save train_encodings.npz
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "train_encodings.npz"
        np.savez(output_path, embeddings=train_encs, cats=train_cats)
        print(f"✅ Saved train_encodings.npz to {output_path}")
        print(f"   Size: {train_encs.nbytes / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Error saving train_encodings.npz: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--output-dir", help="Path to save train_encodings.npz (defaults to model-dir)")
    args = parser.parse_args()
    
    success = generate_train_encodings(args.model_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
