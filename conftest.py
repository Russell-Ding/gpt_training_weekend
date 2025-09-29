"""
Pytest configuration and common fixtures for PyTorch model testing.
"""

import gc
import os
import random
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture(scope="session")
def device() -> torch.device:
    """
    Fixture to determine and return the best available device.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    return device


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """
    Fixture that always returns CPU device for CPU-specific tests.

    Returns:
        torch.device: CPU device
    """
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    """
    Fixture that returns GPU device, skipping test if GPU not available.

    Returns:
        torch.device: CUDA device
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return torch.device("cuda")


@pytest.fixture(scope="function")
def seed_everything():
    """
    Fixture to set random seeds for reproducible tests.
    Call this fixture in tests that need deterministic behavior.
    """
    seed = 42

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

    yield seed

    # Reset deterministic settings after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture(scope="function")
def cleanup_memory():
    """
    Fixture to clean up GPU memory before and after tests.
    """
    # Clear cache before test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()

    yield

    # Clear cache after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Fixture that provides a temporary directory for test files.

    Yields:
        Path: Temporary directory path
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def simple_model(device: torch.device) -> nn.Module:
    """
    Fixture that provides a simple neural network for testing.

    Args:
        device: Device to place the model on

    Returns:
        nn.Module: Simple 2-layer MLP
    """
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)

    return model


@pytest.fixture(scope="function")
def sample_data(device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Fixture that provides sample input data for testing.

    Args:
        device: Device to place tensors on

    Returns:
        Dict[str, torch.Tensor]: Dictionary with sample input and target tensors
    """
    batch_size = 8
    input_size = 10
    num_classes = 5

    inputs = torch.randn(batch_size, input_size, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)

    return {
        "inputs": inputs,
        "targets": targets
    }


@pytest.fixture(scope="function")
def optimizer_config() -> Dict[str, Any]:
    """
    Fixture that provides common optimizer configurations.

    Returns:
        Dict[str, Any]: Optimizer configuration
    """
    return {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999)
    }


@pytest.fixture(scope="function")
def training_config() -> Dict[str, Any]:
    """
    Fixture that provides common training configurations.

    Returns:
        Dict[str, Any]: Training configuration
    """
    return {
        "epochs": 5,
        "batch_size": 16,
        "log_interval": 10,
        "save_interval": 100
    }


@pytest.fixture(scope="session")
def model_checkpoint_dir(tmp_path_factory) -> Path:
    """
    Session-scoped fixture for model checkpoint directory.

    Returns:
        Path: Directory for saving model checkpoints during tests
    """
    checkpoint_dir = tmp_path_factory.mktemp("model_checkpoints")
    return checkpoint_dir


@pytest.fixture(scope="function")
def mock_dataset_config() -> Dict[str, Any]:
    """
    Fixture that provides mock dataset configuration.

    Returns:
        Dict[str, Any]: Dataset configuration
    """
    return {
        "num_samples": 100,
        "input_dim": 10,
        "num_classes": 5,
        "noise_level": 0.1
    }


@pytest.fixture(scope="function", autouse=True)
def reset_torch_state():
    """
    Auto-use fixture that resets PyTorch state between tests.
    This helps ensure test isolation.
    """
    yield

    # Reset random number generator state
    if hasattr(torch, '_C') and hasattr(torch._C, '_reset_rng_state'):
        torch._C._reset_rng_state()

    # Clear any cached compilation
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'reset'):
        torch._dynamo.reset()


@pytest.fixture(scope="function")
def memory_monitor():
    """
    Fixture to monitor memory usage during tests.
    Useful for detecting memory leaks.
    """
    initial_memory = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        if initial_memory is not None:
            memory_diff = final_memory - initial_memory
            if memory_diff > 100 * 1024 * 1024:  # 100MB threshold
                print(f"Warning: Memory usage increased by {memory_diff / 1024 / 1024:.2f} MB")


def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Set environment variables for better PyTorch performance during testing
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    # Disable CUDA caching allocator for more accurate memory testing
    if torch.cuda.is_available():
        os.environ.setdefault('PYTORCH_NO_CUDA_MEMORY_CACHING', '1')


def pytest_runtest_setup(item):
    """
    Setup run before each test item.
    """
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # Skip slow tests in fast test runs
    if item.get_closest_marker("slow") and item.config.getoption("-k", default="").find("not slow") != -1:
        pytest.skip("Skipping slow test")


def pytest_addoption(parser):
    """
    Add custom command line options.
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Run only GPU tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify collected test items based on command line options.
    """
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

    if config.getoption("--gpu-only"):
        skip_cpu = pytest.mark.skip(reason="--gpu-only specified")
        for item in items:
            if "gpu" not in item.keywords:
                item.add_marker(skip_cpu)