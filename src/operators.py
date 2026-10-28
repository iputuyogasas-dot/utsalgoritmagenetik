"""Operator genetika (ditulis sendiri, tanpa DEAP/PyGAD).

Semua operator menerima ``numpy.random.Generator`` agar hasil reproducible dan
tidak pernah memodifikasi induk (selalu bekerja pada salinan).
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ inisialisasi
def inisialisasi_populasi(rng: np.random.Generator, pop_size: int, n_gen: int, max_gen: int) -> np.ndarray:
    """Populasi acak ``(pop_size, n_gen)``; tiap gen seragam dari ``[0, max_gen]``."""
    return rng.integers(0, max_gen + 1, size=(pop_size, n_gen))


# ---------------------------------------------------------------------- seleksi
def seleksi_roulette(rng: np.random.Generator, pop: np.ndarray, fit: np.ndarray, n: int = 1) -> np.ndarray:
    """Roulette wheel: peluang terpilih proporsional dengan fitness.

    Mengembalikan salinan ``n`` individu, shape ``(n, N)``. Jika total fitness
    tidak positif, seleksi jatuh ke seragam.
    """
    fit = np.asarray(fit, dtype=float)
    total = fit.sum()
    p = fit / total if total > 0 else np.full(len(fit), 1.0 / len(fit))
    idx = rng.choice(len(pop), size=n, p=p)
    return pop[idx].copy()


def seleksi_turnamen(
    rng: np.random.Generator, pop: np.ndarray, fit: np.ndarray, n: int = 1, k: int = 3
) -> np.ndarray:
    """Turnamen ukuran ``k``: ambil ``k`` individu acak, menangkan fitness tertinggi."""
    fit = np.asarray(fit)
    kandidat = rng.integers(0, len(pop), size=(n, k))
    pemenang = kandidat[np.arange(n), np.argmax(fit[kandidat], axis=1)]
    return pop[pemenang].copy()


def seleksi(metode: str, rng: np.random.Generator, pop: np.ndarray, fit: np.ndarray, n: int, k: int = 3) -> np.ndarray:
    """Pilih ``n`` individu dengan metode ``'roulette'`` atau ``'turnamen'``."""
    if metode == "roulette":
        return seleksi_roulette(rng, pop, fit, n)
    if metode == "turnamen":
        return seleksi_turnamen(rng, pop, fit, n, k)
    raise ValueError(f"Metode seleksi tidak dikenal: {metode!r} (roulette/turnamen)")


# ---------------------------------------------------------------------- crossover
def crossover_satu_titik(
    rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, pc: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crossover satu titik: dengan peluang ``pc`` tukar ekor pada titik potong acak."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < pc and len(p1) > 1:
        titik = int(rng.integers(1, len(p1)))
        c1[titik:], c2[titik:] = p2[titik:].copy(), p1[titik:].copy()
    return c1, c2


def crossover_uniform(
    rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, pc: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crossover uniform: dengan peluang ``pc`` tiap gen ditukar dengan peluang 0,5."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < pc:
        tukar = rng.random(len(p1)) < 0.5
        c1[tukar], c2[tukar] = p2[tukar], p1[tukar]
    return c1, c2


def crossover(
    metode: str, rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, pc: float
) -> tuple[np.ndarray, np.ndarray]:
    """Terapkan crossover ``'satu_titik'`` atau ``'uniform'``."""
    if metode == "satu_titik":
        return crossover_satu_titik(rng, p1, p2, pc)
    if metode == "uniform":
        return crossover_uniform(rng, p1, p2, pc)
    raise ValueError(f"Metode crossover tidak dikenal: {metode!r} (satu_titik/uniform)")


# ------------------------------------------------------------------------ mutasi
def mutasi(rng: np.random.Generator, kromosom: np.ndarray, pm: float, max_gen: int) -> np.ndarray:
    """Mutasi acak per gen: dengan peluang ``pm`` ganti dengan nilai acak ``[0, max_gen]``."""
    anak = kromosom.copy()
    mask = rng.random(len(anak)) < pm
    anak[mask] = rng.integers(0, max_gen + 1, size=int(mask.sum()))
    return anak


# ---------------------------------------------------------------------- elitisme
def elitisme(pop: np.ndarray, fit: np.ndarray, n_elite: int) -> np.ndarray:
    """Salinan ``n_elite`` individu terbaik (fitness tertinggi) untuk generasi berikutnya."""
    if n_elite <= 0:
        return np.empty((0, pop.shape[1]), dtype=pop.dtype)
    idx = np.argsort(-np.asarray(fit), kind="stable")[:n_elite]
    return pop[idx].copy()
