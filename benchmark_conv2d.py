import torch
import torch.nn as nn
import time
import numpy as np
from psthree.conv import Conv2d, FasterConv2d
import matplotlib.pyplot as plt

def benchmark_conv2d(batch_sizes=[1, 8, 16, 32], input_size=28, in_channels=1, out_channels=16, kernel_size=3):
    
    results = {
        'Custom Conv2d': [],
        'Faster Conv2d': [],
        'PyTorch Conv2d': []
    }
    
    custom_conv = Conv2d(in_channels, out_channels, kernel_size)
    faster_conv = FasterConv2d(in_channels, out_channels, kernel_size)
    torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    with torch.no_grad():
        torch_conv.weight.data = custom_conv.filters.data.clone()
        torch_conv.bias.data = custom_conv.biases.data.clone()
        faster_conv.filters.data = custom_conv.filters.data.clone()
        faster_conv.biases.data = custom_conv.biases.data.clone()

    num_runs = 5
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        x = torch.randn(batch_size, in_channels, input_size, input_size)
        x_torch = torch.tensor(x, dtype=torch.float32)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = custom_conv.forward(x)
            times.append(time.time() - start)
        results['Custom Conv2d'].append(np.mean(times))
        print(f"Custom Conv2d: {results['Custom Conv2d'][-1]:.4f} seconds")
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = faster_conv.forward(x)
            times.append(time.time() - start)
        results['Faster Conv2d'].append(np.mean(times))
        print(f"Faster Conv2d: {results['Faster Conv2d'][-1]:.4f} seconds")
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = torch_conv(x_torch)
            times.append(time.time() - start)
        results['PyTorch Conv2d'].append(np.mean(times))
        print(f"PyTorch Conv2d: {results['PyTorch Conv2d'][-1]:.4f} seconds")
    
    return results, batch_sizes

def plot_results(results, batch_sizes):
    """Plot the benchmark results."""
    plt.figure(figsize=(10, 6))
    
    for implementation, times in results.items():
        plt.plot(batch_sizes, times, marker='o', label=implementation)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Conv2d Implementation Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('conv2d_benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    batch_sizes = [1, 4, 8, 16]
    results, batch_sizes = benchmark_conv2d(batch_sizes=batch_sizes)
    
    print("\nSpeedup Metrics:")
    for i, batch_size in enumerate(batch_sizes):
        custom_time = results['Custom Conv2d'][i]
        faster_time = results['Faster Conv2d'][i]
        pytorch_time = results['PyTorch Conv2d'][i]
        
        print(f"\nBatch size {batch_size}:")
        print(f"Faster vs Custom speedup: {custom_time/faster_time:.2f}x")
        print(f"PyTorch vs Custom speedup: {custom_time/pytorch_time:.2f}x")
        print(f"PyTorch vs Faster speedup: {faster_time/pytorch_time:.2f}x") 