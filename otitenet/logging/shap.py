import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os
import torch.nn as nn
import pickle

CHANNEL_MEAN = [0.13854676, 0.10721603, 0.09241733]
CHANNEL_STD = [0.07892648, 0.07227526, 0.06690206]


def _get_embeddings_cache_path(log_path, group, layer=None):
    """Get the cache file path for embeddings."""
    os.makedirs(f'{log_path}/.cache', exist_ok=True)
    if layer is not None:
        return f'{log_path}/.cache/embeddings_{group}_layer{layer}.pkl'
    return f'{log_path}/.cache/embeddings_{group}.pkl'


def _load_or_compute_embeddings(images, model, log_path, group, layer=None, device='cuda', max_batch=32):
    """
    Load cached embeddings or compute and cache them.
    
    Args:
        images: torch tensor of images
        model: CNN model for encoding
        log_path: path to cache embeddings
        group: 'train' or 'queries'
        layer: optional layer index for intermediate embeddings
        device: 'cuda' or 'cpu'
        max_batch: batch size for encoding (to avoid OOM)
    
    Returns:
        embeddings: numpy array of shape (n_images, embedding_dim)
    """
    cache_path = _get_embeddings_cache_path(log_path, group, layer)
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"[CACHE] Loading embeddings from {cache_path}...")
        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"[CACHE] ✓ Loaded {embeddings.shape[0]} embeddings of shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"[CACHE] Failed to load: {e}, recomputing...")
    
    # Compute embeddings
    print(f"[CACHE] Computing embeddings for {group} ({images.shape[0]} images)...")
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for idx in range(0, images.shape[0], max_batch):
            batch = images[idx:idx+max_batch].to(device)
            if layer is not None:
                # Get intermediate layer output
                try:
                    emb = model.model[layer](batch)
                except:
                    emb = model.model.encoder.layers[layer](batch)
            else:
                # Get final embedding
                emb = model.model(batch) if hasattr(model, 'model') else model(batch)
            
            embeddings_list.append(emb.detach().cpu().numpy())
            print(f"[CACHE]   {idx + batch.shape[0]}/{images.shape[0]}")
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Reshape if needed (remove spatial dims if present)
    if len(embeddings.shape) > 2:
        original_shape = embeddings.shape
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        print(f"[CACHE] Reshaped from {original_shape} to {embeddings.shape}")
    
    # Cache the embeddings
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"[CACHE] ✓ Cached {embeddings.shape[0]} embeddings to {cache_path}")
    except Exception as e:
        print(f"[CACHE] Warning: Could not cache embeddings: {e}")
    
    return embeddings


def _images_display(images: torch.Tensor) -> np.ndarray:
    """Return image batch in HWC np format for plotting."""
    return denormalize_for_display(images).detach().cpu().numpy().transpose(0, 2, 3, 1)


def denormalize_for_display(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CHANNEL_MEAN, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(CHANNEL_STD, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)

def remove_batchnorm(module):
    if isinstance(module, nn.BatchNorm2d):
        return nn.Identity()  # Replace BatchNorm with Identity (no-op)
    for name, child in module.named_children():
        module.add_module(name, remove_batchnorm(child))
    return module

def replace_bn_with_gn(module, num_groups=32):
    """Recursively replace all BatchNorm2d layers with GroupNorm in a model."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features  # Get the number of channels
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))  # Replace with GroupNorm
        else:
            replace_bn_with_gn(child, num_groups)  # Recursively process submodules
    return module


def get_gradient_explaination(i, nets, images, group, log_path, name, layer=5):
    """
    Gets the gradient explaination of the images
    Args:
        i: index of the image
        run: Neptune run
        nets: models
        images: images
        group: group of the image
        log_path: path to save the images
        name: name of the image
    """
    try:
        explainer = shap.GradientExplainer(
            (nets['cnn'].model, nets['cnn'].model[layer]),
            images
        )
    except Exception as e:
        explainer = shap.GradientExplainer(
            (nets['cnn'].model, nets['cnn'].model.encoder.layers[layer]),
            images
        )
    

    shap_values, indexes = explainer.shap_values(images[i:i+1], ranked_outputs=2, nsamples=200)
    images_display = _images_display(images)
    try:
        shap_values = [s.transpose(3, 1, 2, 0) for s in shap_values]
    except Exception:
        pass
    shap.image_plot(shap_values, images_display[i:i+1])
    # shap.image_plot(shap_values)
    f_emb = plt.gcf()
    plt.savefig(f'{log_path}/gradients_shap/{group}_{name}_layer{layer}.png')
    plt.close(f_emb)

def get_explanation_with_knn(i, nets, inputs, group, log_path, name, device='cuda'):
    """
    Gets the explanation of the images with cached embeddings (no layer decomposition)
    Args:
        i: index of the image
        nets: models
        inputs: input batch
        group: group of the image
        log_path: path to save the images
        name: name of the image
    """
    import time
    print(f"\n[SHAP] Starting get_explanation_with_knn for {name}")
    start_time = time.time()
    
    images = torch.concatenate(inputs[group]['inputs'])
    images_display = _images_display(images)
    images = images.to(device)
    images_train = torch.concatenate(inputs['train']['inputs'])
    images_train = images_train.to(device)
    nets['cnn'].eval()

    # Load or compute cached embeddings
    print(f"[SHAP] Loading/computing background embeddings...")
    bg_embeddings = _load_or_compute_embeddings(
        images_train, nets['cnn'], log_path, 'train_bg', layer=None, device=device
    )
    
    background_samples = min(10, bg_embeddings.shape[0])
    print(f"[SHAP] Creating KernelExplainer with {background_samples} samples...")
    explainer_knn = shap.KernelExplainer(
        nets['knn'].predict_proba,
        bg_embeddings[:background_samples]
    )
    
    # Get test embedding
    test_embedding = _load_or_compute_embeddings(
        images[i:i+1], nets['cnn'], log_path, f'{group}_test', layer=None, device=device
    )
    
    print(f"[SHAP] Computing KNN SHAP values ({time.time()-start_time:.1f}s)...")
    shap_values_knn = explainer_knn.shap_values(test_embedding)[0]  # Shape: (num_classes, 1, embedding_dim)
    print(f"[SHAP] KNN SHAP done ({time.time()-start_time:.1f}s)")

    # SHAP for CNN embeddings
    print(f"[SHAP] Creating GradientExplainer...")
    try:
        explainer_cnn = shap.GradientExplainer(
            nets['cnn'].model, torch.tensor(images).to(device)
        )  # Background images
        
        # Get SHAP values for the test image
        test_image = torch.tensor(
            inputs[group]['inputs'][0][i:i+1]
        ).to(device)  # Shape: (1, C, H, W)
        print(f"[SHAP] test_image shape: {test_image.shape}")
        
        print(f"[SHAP] Computing gradient SHAP ({time.time()-start_time:.1f}s)...")
        shap_values_cnn = explainer_cnn.shap_values(test_image)  # (C, H, W, embedding_dim)
        shap_values_cnn = np.array(shap_values_cnn)  # Convert to NumPy
        print(f"[SHAP] Gradient SHAP done ({time.time()-start_time:.1f}s)")
        
        # Multiply signed SHAP values
        final_shap_values = [
            np.sum(shap_values_cnn * shap_values_knn[None, None, None, :, i], axis=-1) 
            for i in range(shap_values_knn.shape[1])
        ]  # (C, H, W)
        final_shap_values = [x.transpose(0, 2, 3, 1) for x in final_shap_values]
    except Exception as e:
        print(f"[SHAP] Gradient explainer failed: {e}, using KNN only...")
        final_shap_values = [x.transpose(1, 2, 0) for x in shap_values_knn]
    
    print(f"[SHAP] Creating image plot...")
    shap.image_plot(final_shap_values, images_display[i:i+1])
    f_emb2 = plt.gcf()
    
    # Ensure directory exists
    save_dir = os.path.dirname(f'{log_path}/knn_shap/{group}_{name}.png')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = f'{log_path}/knn_shap/{group}_{name}.png'
    print(f"[SHAP] Saving to {save_path}...")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(f_emb2)
    print(f"[SHAP] ✓ Complete! ({time.time()-start_time:.1f}s total)\n")

def get_explanation_layer_with_knn(i, nets, inputs, group, log_path, name, device='cuda', layer=5):
    """
    Gets the explanation of the images with cached embeddings
    Args:
        i: index of the image
        nets: models
        inputs: input batch
        group: group of the image
        log_path: path to save the images
        name: name of the image
        layer: layer index for gradient SHAP
    """
    import time
    print(f"\n[SHAP] Starting get_explanation_layer_with_knn for {name}, layer={layer}")
    start_time = time.time()
    
    images = torch.concatenate(inputs[group]['inputs'])
    images_display = _images_display(images)
    images = images.to(device)
    images_train = torch.concatenate(inputs['train']['inputs'])
    images_train = images_train.to(device)
    nets['cnn'].eval()

    # Load or compute cached embeddings for background training set
    print(f"[SHAP] Loading/computing background embeddings...")
    bg_embeddings = _load_or_compute_embeddings(
        images_train, nets['cnn'], log_path, 'train_bg', layer=None, device=device
    )
    
    background_samples = min(10, bg_embeddings.shape[0])
    print(f"[SHAP] Creating KernelExplainer with {background_samples} samples...")
    explainer_knn = shap.KernelExplainer(
        nets['knn'].predict_proba,
        bg_embeddings[:background_samples]
    )
    
    # Get test embedding (usually just 1 image)
    test_embedding = _load_or_compute_embeddings(
        images[i:i+1], nets['cnn'], log_path, f'{group}_test', layer=None, device=device
    )
    
    print(f"[SHAP] Computing KNN SHAP values ({time.time()-start_time:.1f}s)...")
    shap_values_knn = explainer_knn.shap_values(test_embedding)[0]  # Shape: (num_classes, 1, embedding_dim)
    print(f"[SHAP] KNN SHAP done ({time.time()-start_time:.1f}s)")

    # Get SHAP values for the predicted class
    # SHAP for CNN embeddings
    print(f"[SHAP] Creating GradientExplainer for layer {layer}...")
    try:
        explainer_cnn = shap.GradientExplainer(
            (nets['cnn'].model, nets['cnn'].model[layer]),
            torch.tensor(images).to(device)
        )  # Background images
    except Exception as e:
        print(f"[SHAP] Layer {layer} indexing failed: {e}, trying .encoder.layers...")
        explainer_cnn = shap.GradientExplainer(
            (nets['cnn'].model, nets['cnn'].model.encoder.layers[layer]),
            torch.tensor(images).to(device)
        )

    # Get SHAP values for the test image
    test_image = torch.tensor(
        inputs[group]['inputs'][0][i:i+1]
    ).to(device)  # Shape: (1, C, H, W)
    print(f"[SHAP] test_image shape: {test_image.shape}")
    
    print(f"[SHAP] Computing gradient SHAP ({time.time()-start_time:.1f}s)...")
    shap_values_cnn = explainer_cnn.shap_values(test_image)  # (C, H, W, embedding_dim)
    shap_values_cnn = np.array(shap_values_cnn)  # Convert to NumPy
    print(f"[SHAP] Gradient SHAP done ({time.time()-start_time:.1f}s)")

    # Multiply signed SHAP values (retain positive and negative contributions)
    final_shap_values = [
        np.sum(shap_values_cnn * shap_values_knn[None, None, None, :, i], axis=-1) 
        for i in range(shap_values_knn.shape[1])
    ]  # (C, H, W)
    final_shap_values = [x.transpose(0, 2, 3, 1) for x in final_shap_values]
    
    print(f"[SHAP] Creating image plot...")
    shap.image_plot(final_shap_values, images_display[i:i+1])
    f_emb2 = plt.gcf()
    
    # Ensure directory exists
    save_dir = os.path.dirname(f'{log_path}/knn_shap/{group}_{name}_layer{layer}.png')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = f'{log_path}/knn_shap/{group}_{name}_layer{layer}.png'
    print(f"[SHAP] Saving to {save_path}...")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(f_emb2)
    print(f"[SHAP] ✓ Complete! ({time.time()-start_time:.1f}s total)\n")

def get_explanation_deep(i, nets, inputs, group, log_path, name, bs, device):
    """
    Gets the explaination of the images
    Args:
        i: index of the image
        nets: models
        images: images
        group: group of the image
        log_path: path to save the images
        name: name of the image
    """
    # images = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs[group]['inputs']])
    images = torch.concatenate(inputs[group]['inputs'])
    images_display = _images_display(images)
    images = images.to(device)
    # images_train = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs['train']['inputs']])
    images_train = torch.concatenate(inputs['train']['inputs'])
    # images_train = transform(images_train)
    images_train = images_train.to(device)
    nets['cnn'].eval()

    # model = remove_batchnorm(nets['cnn'].model)
    model = replace_bn_with_gn(nets['cnn'].model)
    model = nets['cnn'].model.to(device)
    # SHAP on k-NN (signed values)
    explainer_knn = shap.DeepExplainer(
        model,
        images_train.squeeze().detach()[:bs]
    )
    # Small background set
    test_embedding = model(images[i:i+1]).reshape(1, -1)
    shap_values_knn = explainer_knn.shap_values(images[i:i+1])[0]  # Shape: (num_classes, 1, embedding_dim)

    shap.image_plot(shap_values_knn, images_display[i:i+1])
    # shap.image_plot(shap_values)
    f_emb2 = plt.gcf()
    plt.savefig(f'{log_path}/knn_shap/{group}_{name}_deepexplainer.png')
    plt.close(f_emb2)

def get_explanation_deep_kernel(i, nets, inputs, group, log_path, name, device):
    """
    Gets the explaination of the images
    Args:
        i: index of the image
        nets: models
        images: images
        group: group of the image
        log_path: path to save the images
        name: name of the image
    """
    # images = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs[group]['inputs']])
    images = torch.concatenate(inputs[group]['inputs'])
    images_display = _images_display(images)
    images = images.to(device)
    # images_train = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs['train']['inputs']])
    images_train = torch.concatenate(inputs['train']['inputs'])
    # images_train = transform(images_train)
    images_train = images_train.to(device)
    nets['cnn'].eval()

    def remove_batchnorm(module):
        if isinstance(module, nn.BatchNorm2d):
            return nn.Identity()  # Replace BatchNorm with Identity (no-op)
        for name, child in module.named_children():
            module.add_module(name, remove_batchnorm(child))
        return module


    # model = remove_batchnorm(nets['cnn'].model)
    # model = replace_bn_with_gn(nets['cnn'].model)
    nets['cnn'].model.to('cpu')
    nets['cnn'].linear.to('cpu')
    # SHAP on k-NN (signed values)
    explainer_knn = shap.KernelExplainer(
        nets['cnn'].predict_proba,
        images_train[:5].detach().cpu().numpy().reshape(5, -1)
    )
    # Small background set
    try:
        shap_values_knn = explainer_knn.shap_values(images[i:i+1].detach().cpu().numpy().reshape(1, -1))[0]  # Shape: (num_classes, 1, embedding_dim)
    except:
        return

    shap.image_plot(shap_values_knn, images_display[i:i+1])
    # shap.image_plot(shap_values)
    f_emb2 = plt.gcf()
    plt.savefig(f'{log_path}/knn_shap/{group}_{name}_deepkernel.png')
    plt.close(f_emb2)

def get_explanation_layer(i, nets, inputs, group, log_path, name, device, layer=5):
    """
    Gets the explaination of the images
    Args:
        i: index of the image
        nets: models
        images: images
        group: group of the image
        log_path: path to save the images
        name: name of the image
    """
    # images = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs[group]['inputs']])
    images = torch.concatenate(inputs[group]['inputs'])
    images_display = _images_display(images)
    images = images.to(device)
    # images_train = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs['train']['inputs']])
    images_train = torch.concatenate(inputs['train']['inputs'])
    images_train = images_train.to(device)
    nets['cnn'].eval()

    # SHAP on k-NN (signed values)
    nets['cnn'] = replace_bn_with_gn(nets['cnn'].model)
    nets['cnn'] = nets['cnn'].model.to(device)
    explainer_knn = shap.KernelExplainer(
        nets['cnn'].predict_proba,
        images_train[:50].squeeze().detach().cpu().numpy()
    )
    # Small background set
    test_embedding = nets['cnn'](images[i:i+1]).detach().cpu().numpy().reshape(1, -1)
    shap_values_knn = explainer_knn.shap_values(test_embedding)[0]  # Shape: (num_classes, 1, embedding_dim)

    # SHAP for CNN embeddings
    explainer_cnn = shap.GradientExplainer((nets['cnn'].model, nets['cnn'].model[layer]), torch.tensor(images).to(device))  # Background images

    # Get SHAP values for the test image
    test_image = torch.tensor(inputs[group]['inputs'][0][i:i+1]).to(device)  # Shape: (1, C, H, W)
    print('test_image', test_image.shape)
    shap_values_cnn = explainer_cnn.shap_values(test_image)  # (C, H, W, embedding_dim)
    shap_values_cnn = np.array(shap_values_cnn)  # Convert to NumPy

    # Multiply signed SHAP values (retain positive and negative contributions)
    final_shap_values = [np.sum(shap_values_cnn * shap_values_knn[None, None, None, :, i], axis=-1) for i in range(shap_values_knn.shape[1])]  # (C, H, W)
    final_shap_values = [x.transpose(0, 2, 3, 1) for x in final_shap_values]
    shap.image_plot(final_shap_values, images_display[i:i+1])
    # shap.image_plot(shap_values)
    f_emb2 = plt.gcf()
    plt.savefig(f'{log_path}/cnn_shap/{group}_{name}_layer{layer}.png')
    plt.close(f_emb2)

def log_shap_images_gradients(nets, i, inputs, group, log_path, name, device='cuda'):
    """
    Logs SHAP values and gradients of the images
    Args:
        run: Neptune run
        nets: models
        i: index of the image
        inputs: inputs
        group: group of the image
        log_path: path to save the images
        name: name of the image
        mlops: 'neptune' or 'mlflow'
    """

    images = torch.concatenate([torch.Tensor(x) for x in inputs[group]['inputs']]).to(device)
    
    # images = torch.stack([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in images])
    # images = images.reshape(-1, 224, 224, 3)
    nets['cnn'].eval()
    save_dir = os.path.dirname(f'{log_path}/knn_shap/{group}_{name}_layer5.png')
    os.makedirs(save_dir, exist_ok=True)
    get_gradient_explaination(i, nets, images, group, log_path, name)
    if not isinstance(nets['knn'], nn.Module):
        print('Using KNN model')
        get_explanation_layer_with_knn(i, nets, inputs, group, log_path, name)
        # get_explanation_with_knn(i, nets, inputs, group, log_path, name)
    else:
        print('No KNN model')
        get_explanation_deep(i, nets, inputs, group, log_path, name)
        # get_explanation_with_knn(i, nets, inputs, group, log_path, name)


def log_shap_gradients_only(nets, i, inputs, group, name, log_path, layer=5, device='cuda'):
    """Wrapper used by app.py: compute and save gradient SHAP for a single image."""
    images = torch.concatenate(inputs[group]['inputs']).to(device)
    os.makedirs(f'{log_path}/gradients_shap', exist_ok=True)
    try:
        try:
            explainer = shap.GradientExplainer((nets['cnn'].model, nets['cnn'].model[layer]), images)
        except Exception:
            explainer = shap.GradientExplainer((nets['cnn'].model, nets['cnn'].model.encoder.layers[layer]), images)
        shap_values = explainer.shap_values(images[i:i+1], nsamples=200)
        images_display = _images_display(images)
        shap.image_plot(shap_values, images_display[i:i+1])
        plt.savefig(f'{log_path}/gradients_shap/{group}_{name}_layer{layer}.png')
        plt.close()
    except Exception as e:
        print(f"Error in log_shap_gradients_only: {e}")


def log_shap_knn_or_deep(nets, i, inputs, group, name, log_path, layer=5, device='cuda', bs=16):
    """Wrapper used by app.py: choose KNN- or deep-based SHAP depending on nets['knn']."""
    os.makedirs(f'{log_path}/knn_shap', exist_ok=True)
    try:
        if not isinstance(nets.get('knn'), nn.Module):
            try:
                get_explanation_layer_with_knn(i, nets, inputs, group, log_path, name, device=device, layer=layer)
            except Exception:
                get_explanation_with_knn(i, nets, inputs, group, log_path, name, device=device)
        else:
            try:
                get_explanation_deep(i, nets, inputs, group, log_path, name, bs=bs, device=device)
            except Exception:
                get_explanation_deep_kernel(i, nets, inputs, group, log_path, name, device=device)
    except Exception as e:
        print(f"Error in log_shap_knn_or_deep: {e}")

