"""Representasi kromosom.

Kromosom = array integer panjang N (satu gen per sesi/kelas). Nilai gen
``g in [0, n_slot*n_ruang - 1]`` mengkodekan pasangan (slot, ruang):

    slot  = g // n_ruang
    ruang = g %  n_ruang
"""
from __future__ import annotations

import numpy as np

N_GEN = 24
N_SLOT = 15  # 5 hari x 3 sesi (jam kerja 07.00-17.00)
N_RUANG = 5
MAX_GEN = N_SLOT * N_RUANG - 1  # 74


def decode(g, n_ruang: int = N_RUANG):
    """Dekode gen (skalar atau array) menjadi ``(slot, ruang)``."""
    g = np.asarray(g) if not isinstance(g, (int, np.integer)) else int(g)
    return g // n_ruang, g % n_ruang


def encode(slot, ruang, n_ruang: int = N_RUANG):
    """Enkode ``(slot, ruang)`` menjadi nilai gen."""
    return slot * n_ruang + ruang


def decode_kromosom(kromosom: np.ndarray, n_ruang: int = N_RUANG) -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan array ``slot`` dan ``ruang`` untuk seluruh gen kromosom."""
    kromosom = np.asarray(kromosom)
    return kromosom // n_ruang, kromosom % n_ruang


def kromosom_acak(rng: np.random.Generator, n_gen: int = N_GEN, max_gen: int = MAX_GEN) -> np.ndarray:
    """Buat satu kromosom acak; tiap gen seragam dari ``[0, max_gen]``."""
    return rng.integers(0, max_gen + 1, size=n_gen)


def valid(kromosom: np.ndarray, n_gen: int = N_GEN, max_gen: int = MAX_GEN) -> bool:
    """Cek panjang dan rentang gen."""
    k = np.asarray(kromosom)
    return k.shape == (n_gen,) and bool(np.all((k >= 0) & (k <= max_gen)))
