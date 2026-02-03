from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def device_cpu() -> torch.device:
    return torch.device("cpu")
