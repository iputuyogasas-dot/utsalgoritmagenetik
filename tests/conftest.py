import numpy as np
import pytest

from src.chromosome import encode
from src.data_loader import load_data


@pytest.fixture(scope="session")
def data():
    return load_data()


def taruh(k: np.ndarray, data, idx: int, slot: int, ruang: int) -> None:
    """Letakkan sesi ``idx`` pada (slot, ruang)."""
    k[idx] = encode(slot, ruang, data.n_ruang)


@pytest.fixture
def jadwal_valid(data) -> np.ndarray:
    """Jadwal valid buatan tangan: matkul ke-i (pasangan A/B) di slot i; A di FT-5, B di FT-4."""
    k = np.zeros(data.n_gen, dtype=int)
    for i, (a, b) in enumerate(data.pasangan):
        taruh(k, data, a, i, 4)
        taruh(k, data, b, i, 3)
    return k
