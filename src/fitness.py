"""Fungsi fitness berbasis penalti kendala keras (H1-H5) + soft constraint (informasi).

    total_pelanggaran = H1 + H2 + H3 + H4 + H5
    fitness           = 1 / (1 + total_pelanggaran)

Semua fungsi menerima kromosom 1-D ``(N,)`` atau populasi 2-D ``(P, N)``;
hasilnya skalar atau array ``(P,)``.
"""
from __future__ import annotations

import numpy as np

from .data_loader import DataJadwal

NAMA_KENDALA = ("H1", "H2", "H3", "H4", "H5")


def _as2d(pop: np.ndarray) -> tuple[np.ndarray, bool]:
    pop = np.asarray(pop)
    if pop.ndim == 1:
        return pop[None, :], True
    return pop, False


def _out(hasil: np.ndarray, satu: bool):
    return int(hasil[0]) if satu else hasil


def _bentrok_pasangan(slot: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Jumlah pasangan (i,j) pada ``pairs`` yang slotnya sama, per individu."""
    if len(pairs) == 0:
        return np.zeros(slot.shape[0], dtype=int)
    return (slot[:, pairs[:, 0]] == slot[:, pairs[:, 1]]).sum(axis=1)


def h1_dosen(pop: np.ndarray, data: DataJadwal):
    """H1: jumlah pasangan sesi satu dosen yang berada pada slot sama."""
    p, satu = _as2d(pop)
    return _out(_bentrok_pasangan(p // data.n_ruang, data.pairs_dosen), satu)


def h2_ruang(pop: np.ndarray, data: DataJadwal):
    """H2: jumlah pasangan sesi pada (slot, ruang) yang sama.

    Karena gen = slot*R + ruang, dua sesi pada slot dan ruang sama iff gennya sama.
    """
    p, satu = _as2d(pop)
    pairs = data.pairs_all
    return _out((p[:, pairs[:, 0]] == p[:, pairs[:, 1]]).sum(axis=1), satu)


def h3_kelompok(pop: np.ndarray, data: DataJadwal):
    """H3: jumlah pasangan sesi satu kelompok mahasiswa pada slot sama."""
    p, satu = _as2d(pop)
    return _out(_bentrok_pasangan(p // data.n_ruang, data.pairs_kelompok), satu)


def h4_kapasitas(pop: np.ndarray, data: DataJadwal):
    """H4: jumlah sesi yang ditempatkan di ruang berkapasitas kurang."""
    p, satu = _as2d(pop)
    kap = data.kapasitas[p % data.n_ruang]
    return _out((kap < data.jumlah_mhs[None, :]).sum(axis=1), satu)


def h5_paralel(pop: np.ndarray, data: DataJadwal, paralel: bool = True):
    """H5: jumlah pasangan kelas A/B yang slot waktunya berbeda (maks 12).

    Jika ``paralel=False`` kendala dinonaktifkan (selalu 0).
    """
    p, satu = _as2d(pop)
    if not paralel:
        return _out(np.zeros(p.shape[0], dtype=int), satu)
    slot = p // data.n_ruang
    return _out((slot[:, data.pair_a] != slot[:, data.pair_b]).sum(axis=1), satu)


def pelanggaran(pop: np.ndarray, data: DataJadwal, paralel: bool = True) -> dict:
    """Rincian pelanggaran ``{'H1':..., ..., 'H5':...}`` (nilai skalar/array)."""
    return {
        "H1": h1_dosen(pop, data),
        "H2": h2_ruang(pop, data),
        "H3": h3_kelompok(pop, data),
        "H4": h4_kapasitas(pop, data),
        "H5": h5_paralel(pop, data, paralel),
    }


def total_pelanggaran(pop: np.ndarray, data: DataJadwal, paralel: bool = True):
    """Jumlah seluruh pelanggaran H1..H5."""
    r = pelanggaran(pop, data, paralel)
    return sum(r[k] for k in NAMA_KENDALA)


def fitness(pop: np.ndarray, data: DataJadwal, paralel: bool = True):
    """``1 / (1 + total_pelanggaran)``; 1.0 iff jadwal tanpa bentrok."""
    return 1.0 / (1.0 + total_pelanggaran(pop, data, paralel))


def evaluasi(kromosom: np.ndarray, data: DataJadwal, paralel: bool = True) -> tuple[float, dict]:
    """Fitness satu kromosom beserta rincian pelanggaran H1-H5."""
    r = pelanggaran(kromosom, data, paralel)
    return 1.0 / (1.0 + sum(r.values())), r


# ---------------------------------------------------------------- soft constraint
def s1_preferensi_dosen(kromosom: np.ndarray, data: DataJadwal) -> int:
    """S1: jumlah sesi yang jatuh pada slot yang tidak diinginkan dosennya."""
    slot = np.asarray(kromosom) // data.n_ruang
    return sum(
        1 for s, sl in zip(data.sesi, slot) if int(sl) in data.pref_dosen.get(s.dosen, set())
    )


def s2_jeda_kelompok(kromosom: np.ndarray, data: DataJadwal) -> int:
    """S2: total jeda (jumlah sesi kosong) antar kuliah satu kelompok pada hari yang sama."""
    slot = np.asarray(kromosom) // data.n_ruang
    per_hari: dict[tuple[int, int], list[int]] = {}
    for s_idx, sl in enumerate(slot):
        sl = int(sl)
        hari, sesi = int(data.slot_hari[sl]), int(data.slot_sesi[sl])
        per_hari.setdefault((int(data.kelompok_id[s_idx]), hari), []).append(sesi)
    total = 0
    for sesi_list in per_hari.values():
        u = sorted(set(sesi_list))
        total += sum(b - a - 1 for a, b in zip(u, u[1:]))
    return total


def soft_constraint(kromosom: np.ndarray, data: DataJadwal) -> dict:
    """Laporan soft constraint (informasi saja; tidak masuk fitness)."""
    return {
        "S1_preferensi_dosen": s1_preferensi_dosen(kromosom, data),
        "S2_jeda_kelompok": s2_jeda_kelompok(kromosom, data),
    }
