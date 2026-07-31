import pytest
from pathlib import Path

from src.data.loader import AdMIReRepository

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw" / "admire2_data"
TEMPLATES_ROOT = Path(__file__).parent.parent / "data" / "submissions" / "templates"


@pytest.fixture(scope="session")
def repo() -> AdMIReRepository:
    return AdMIReRepository(
        data_root=DATA_ROOT,
        templates_root=TEMPLATES_ROOT,
    )
