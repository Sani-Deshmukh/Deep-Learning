import torch
import pytest
import numpy as np
import time
from pathlib import Path
from psthree.conv import Conv2d, FasterConv2d
from psthree.utils import seed_everything

seed_everything()

def generate_test_data(name):
    test_data_dir = Path(__file__).parent / "test_data"
    save_path = test_data_dir / f"{name}.npz"

    x = torch.ones(1, 10, 10, 8)
    filters = torch.ones(3, 10, 5, 5)
    biases = torch.ones(3)
    padding = 0
    stride = 1
    reference = Conv2d(x.shape[0], filters.shape[0], filters.shape[2], padding, stride, filters, biases)(x)
    reference = reference.detach().numpy()
    np.savez(save_path, x=x, filters=filters, biases=biases, padding=padding, stride=stride, reference=reference)

def load_test_data(name):
    test_data_dir = Path(__file__).parent / "test_data"
    load_path = test_data_dir / f"{name}.npz"
    
    if not load_path.exists():
        raise FileNotFoundError(f"Test data file {name}.npz not found")
    
    data = np.load(load_path)
    return (
        torch.from_numpy(data['x']),
        torch.from_numpy(data['filters']),
        torch.from_numpy(data['biases']),
        int(data['padding']),
        int(data['stride']),
        torch.from_numpy(data['reference'])
    )

# generate_test_data("simple")

@pytest.mark.parametrize(
    "x, filters, biases, padding, stride, reference",
    [
        pytest.param(
            *load_test_data("simple"),
            id="simple",
        ),
        pytest.param(
            *load_test_data("random_32x32"),
            id="random",
        ),
        pytest.param(
            *load_test_data("random_w_padding"),
            id="random_w_padding",
        ),
        pytest.param(
            *load_test_data("random_w_stride"),
            id="random_w_stride",
        ),
        pytest.param(
            *load_test_data("random_large_w_everything"),
            id="random_large_w_everything",
        ),
    ]
)
def test_Conv2d(x, filters, biases, padding, stride, reference):
    conv_layer = Conv2d(x.shape[0], filters.shape[0], filters.shape[2], padding, stride, filters, biases)
    output = conv_layer(x)
    assert torch.norm(output - reference) < 1e-2

@pytest.mark.parametrize(
    "x, filters, biases, padding, stride, reference",
    [
        pytest.param(
            *load_test_data("simple"),
            id="simple",
        ),
        pytest.param(
            *load_test_data("random_32x32"),
            id="random",
        ),
        pytest.param(
            *load_test_data("random_w_padding"),
            id="random_w_padding",
        ),
        pytest.param(
            *load_test_data("random_w_stride"),
            id="random_w_stride",
        ),
        pytest.param(
            *load_test_data("random_large_w_everything"),
            id="random_large_w_everything",
        ),
    ]
)
def test_FasterConv2d(x, filters, biases, padding, stride, reference):
    conv_layer = FasterConv2d(x.shape[0], filters.shape[0], filters.shape[2], padding, stride, filters, biases)
    output = conv_layer(x)
    assert torch.norm(output - reference) < 1e-2

@pytest.mark.parametrize(
    "x, filters, biases, padding, stride, reference",
    [
        pytest.param(
            *load_test_data("random_large_w_everything"),
            id="random_large_w_everything",
        ),
    ]
)
def test_FasterConv2d_speedup(x, filters, biases, padding, stride, reference):
    # Time Conv2d
    conv_layer = Conv2d(x.shape[1], filters.shape[0], filters.shape[2], padding, stride, filters, biases)
    
    start = time.time()
    _ = conv_layer(x)
    conv_time = time.time() - start

    # Time FasterConv2d
    faster_conv_layer = FasterConv2d(x.shape[1], filters.shape[0], filters.shape[2], padding, stride, filters, biases)
    
    start = time.time()
    _ = faster_conv_layer(x)
    faster_conv_time = time.time() - start
    print(f"Conv2d time: {conv_time}, FasterConv2d time: {faster_conv_time}")

    # FasterConv2d should be at least 2x faster
    assert faster_conv_time*2 < conv_time
    