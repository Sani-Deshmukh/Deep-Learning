import torch
import pytest
import numpy as np
from pathlib import Path
from psthree.attention import SelfAttention, MultiHeadAttention
from psthree.utils import seed_everything

def load_test_data_attn(name):
    test_data_dir = Path(__file__).parent / "test_data"
    load_path = test_data_dir / f"{name}.npz"
    
    if not load_path.exists():
        raise FileNotFoundError(f"Test data file {name}.npz not found")
    
    data = np.load(load_path)
    return (
        torch.from_numpy(data['x']),
        torch.from_numpy(data['d_model']),
        torch.from_numpy(data['reference'])
    )

def load_test_data_mha(name):
    test_data_dir = Path(__file__).parent / "test_data"
    load_path = test_data_dir / f"{name}.npz"
    
    if not load_path.exists():
        raise FileNotFoundError(f"Test data file {name}.npz not found")
    
    data = np.load(load_path)
    return (
        torch.from_numpy(data['x']),
        torch.from_numpy(data['d_model']),
        torch.from_numpy(data['num_heads']),
        torch.from_numpy(data['reference'])
    )

@pytest.mark.parametrize(
    "x, d_model, reference",
    [
        pytest.param(
            *load_test_data_attn("simple_attn"),
            id="simple",
        ),
        pytest.param(
            *load_test_data_attn("random_attn"),
            id="random",
        ),
    ]
)
def test_SelfAttention(x, d_model, reference):
    seed_everything()
    with torch.no_grad():
        self_attention = SelfAttention(d_model)
        output = self_attention(x)
    assert torch.norm(output - reference) < 1e-2

@pytest.mark.parametrize(
    "x, d_model, num_heads, reference",
    [
        pytest.param(
            *load_test_data_mha("simple_mha"),
            id="simple",
        ),
        pytest.param(
            *load_test_data_mha("random_mha"),
            id="random",
        ),
    ]
)
def test_MultiHeadAttention(x, d_model, num_heads, reference):
    seed_everything()
    with torch.no_grad():
        multi_head_attention = MultiHeadAttention(d_model, num_heads)
        output = multi_head_attention(x)
    assert torch.norm(output - reference) < 1e-2


    