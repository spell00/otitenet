import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn

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
    explainer = shap.GradientExplainer((nets['cnn'].model, nets['cnn'].model[layer]), images)

    shap_values, indexes = explainer.shap_values(images[i:i+1], ranked_outputs=2, nsamples=200)
    # shap_values = explainer.shap_values(images[i:i+1], nsamples=200)
    shap_values = [s.transpose(3, 1, 2, 0) for s in shap_values]
    shap.image_plot(shap_values, images[i:i+1].cpu().detach().numpy().transpose(0, 2, 3, 1))
    # shap.image_plot(shap_values)
    f_emb = plt.gcf()
    plt.savefig(f'{log_path}/gradients_shap/{group}_{name}_layer{layer}.png')
    plt.close(f_emb)

def get_explanation_with_knn(i, nets, inputs, group, log_path, name, device='cuda'):
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
    # images = torch.concatenate([
    #     ((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs[group]['inputs']
    # ])
    images = torch.concatenate(inputs[group]['inputs'])
    images = images.to(device)
    # images_train = torch.concatenate([
    #     ((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs['train']['inputs']
    # ])
    images_train = torch.concatenate(inputs['train']['inputs'])
    images_train = images_train.to(device)
    nets['cnn'].eval()

    # SHAP on k-NN (signed values)
    explainer_knn = shap.KernelExplainer(
        nets['knn'].predict_proba,
        nets['cnn'].model(
            images_train[:50]).squeeze().detach().cpu().numpy()
        )  # Small background set
    test_embedding = nets['cnn'].model(
        images[i:i+1]
    ).detach().cpu().numpy().reshape(1, -1)
    shap_values_knn = explainer_knn.shap_values(test_embedding)[0]  # Shape: (num_classes, 1, embedding_dim)

    # SHAP for CNN embeddings
    explainer_cnn = shap.GradientExplainer(
        nets['cnn'].model, torch.tensor(images).to(device)
    )  # Background images

    # Get SHAP values for the test image
    test_image = torch.tensor(
        inputs[group]['inputs'][0][i:i+1]
    ).to(device)  # Shape: (1, C, H, W)
    print('test_image', test_image.shape)
    shap_values_cnn = explainer_cnn.shap_values(test_image)  # (C, H, W, embedding_dim)
    shap_values_cnn = np.array(shap_values_cnn)  # Convert to NumPy

    # Multiply signed SHAP values (retain positive and negative contributions)
    final_shap_values = [
        np.sum(shap_values_cnn * shap_values_knn[None, None, None, :, i], axis=-1) 
        for i in range(shap_values_knn.shape[1])
    ]  # (C, H, W)
    final_shap_values = [x.transpose(0, 2, 3, 1) for x in final_shap_values]
    shap.image_plot(
        final_shap_values, 
        images[i:i+1].cpu().detach().numpy().transpose(0, 2, 3, 1)
    )
    # shap.image_plot(shap_values)
    f_emb2 = plt.gcf()
    plt.savefig(f'{log_path}/knn_shap/{group}_{name}.png')
    plt.close(f_emb2)

def get_explanation_layer_with_knn(i, nets, inputs, group, log_path, name, device='cuda', layer=5):
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
    images = images.to(device)
    # images_train = torch.concatenate([((x - torch.min(x)) / (torch.max(x) - torch.min(x))) for x in inputs['train']['inputs']])
    images_train = torch.concatenate(inputs['train']['inputs'])
    images_train = images_train.to(device)
    nets['cnn'].eval()

    # SHAP on k-NN (signed values)
    explainer_knn = shap.KernelExplainer(
        nets['knn'].predict_proba,
        nets['cnn'].model(
            images_train[:50]
        ).squeeze().detach().cpu().numpy()
    )
    test_embedding = nets['cnn'](
        images[i:i+1]
    ).detach().cpu().numpy().reshape(1, -1)
    shap_values_knn = explainer_knn.shap_values(test_embedding)[0]  # Shape: (num_classes, 1, embedding_dim)

    # Get SHAP values for the predicted class
    # SHAP for CNN embeddings
    explainer_cnn = shap.GradientExplainer(
        (nets['cnn'].model, nets['cnn'].model[layer]),
        torch.tensor(images).to(device)
    )  # Background images

    # Get SHAP values for the test image
    test_image = torch.tensor(
        inputs[group]['inputs'][0][i:i+1]
    ).to(device)  # Shape: (1, C, H, W)
    print('test_image', test_image.shape)
    shap_values_cnn = explainer_cnn.shap_values(test_image)  # (C, H, W, embedding_dim)
    shap_values_cnn = np.array(shap_values_cnn)  # Convert to NumPy

    # Multiply signed SHAP values (retain positive and negative contributions)
    final_shap_values = [
        np.sum(shap_values_cnn * shap_values_knn[None, None, None, :, i], axis=-1) 
        for i in range(shap_values_knn.shape[1])
    ]  # (C, H, W)
    final_shap_values = [x.transpose(0, 2, 3, 1) for x in final_shap_values]
    shap.image_plot(
        final_shap_values,
        images[i:i+1].cpu().detach().numpy().transpose(0, 2, 3, 1)
    )
    # shap.image_plot(shap_values)
    f_emb2 = plt.gcf()
    plt.savefig(f'{log_path}/knn_shap/{group}_{name}_layer{layer}.png')
    plt.close(f_emb2)

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
    # transform = torchvision.transforms.Normalize([0.13854676, 0.10721603, 0.09241733], [0.07892648, 0.07227526, 0.06690206])
    images = torch.concatenate(inputs[group]['inputs'])
    # images = transform(images)
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

    shap.image_plot(shap_values_knn, images[i:i+1].cpu().detach().numpy().transpose(0, 2, 3, 1))
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
    # transform = torchvision.transforms.Normalize([0.13854676, 0.10721603, 0.09241733], [0.07892648, 0.07227526, 0.06690206])
    images = torch.concatenate(inputs[group]['inputs'])
    # images = transform(images)
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

    shap.image_plot(
        shap_values_knn,
        images[i:i+1].cpu().detach().numpy().transpose(0, 2, 3, 1)
    )
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
    shap.image_plot(final_shap_values, images[i:i+1].cpu().detach().numpy().transpose(0, 2, 3, 1))
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

    get_gradient_explaination(i, nets, images, group, log_path, name)
    if not isinstance(nets['knn'], nn.Module):
        print('Using KNN model')
        get_explanation_layer_with_knn(i, nets, inputs, group, log_path, name)
        # get_explanation_with_knn(i, nets, inputs, group, log_path, name)
    else:
        print('No KNN model')
        get_explanation_deep(i, nets, inputs, group, log_path, name)
        # get_explanation_with_knn(i, nets, inputs, group, log_path, name)

