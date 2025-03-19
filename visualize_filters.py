import torch
import matplotlib.pyplot as plt
import numpy as np
from psthree.cnn import CNN
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import os

def enhance_contrast(img_array, percentile=98):
    vmin, vmax = np.percentile(img_array, [100-percentile, percentile])
    if vmin == vmax:
        vmin = img_array.min()
        vmax = img_array.max()
    if vmin == vmax:
        return img_array
    
    img_scaled = (img_array - vmin) / (vmax - vmin)
    return np.clip(img_scaled, 0, 1)

def load_mnist_samples(n_samples_per_digit=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    samples = []
    for digit in range(10):
        indices = (test_dataset.targets == digit).nonzero().squeeze()
        selected_indices = indices[:n_samples_per_digit]
        for idx in selected_indices:
            img, _ = test_dataset[idx]
            samples.append((img, str(digit)))
    
    return samples

def apply_filter(image, filter_weights):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        
    filter_weights = filter_weights.unsqueeze(0)
    
    result = F.conv2d(image, filter_weights, padding=1)
    
    result = F.relu(result)
    
    return result.squeeze().numpy()

def visualize_filters_and_effects(model_path, n_samples_per_digit=1):
    model = torch.load(model_path, weights_only=False)
    
    filters = model.conv.weight.data
    n_filters = filters.shape[0]
    
    samples = load_mnist_samples(n_samples_per_digit)
    
    n_cols = len(samples) + 1 
    fig, axes = plt.subplots(n_filters, n_cols, figsize=(2*n_cols, 2*n_filters))
    if n_filters == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_filters):
        filter_weights = filters[i, 0].detach().numpy()
        filter_display = enhance_contrast(filter_weights, percentile=98)
        axes[i, 0].imshow(filter_display, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Filter {i+1}')
        
        for j, (img, digit) in enumerate(samples):
            filtered = apply_filter(img, filters[i])
            filtered_display = enhance_contrast(filtered, percentile=98)
            
            axes[i, j+1].imshow(filtered_display, cmap='gray')
            axes[i, j+1].axis('off')
            if i == 0:
                axes[i, j+1].set_title(f'Digit {digit}')

    save_path = os.path.join(os.path.dirname(model_path), "filter_effects.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Filter effects visualization saved at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    visualize_filters_and_effects(args.model_path) 