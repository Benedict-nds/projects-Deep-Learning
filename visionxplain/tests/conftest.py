"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import set_seed


@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def seed():
    """Set random seed for reproducibility."""
    set_seed(42)
    return 42


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch():
    """Create a sample batch."""
    return {
        'images': torch.randn(4, 3, 224, 224),
        'labels': torch.randint(0, 2, (4,))
    }

