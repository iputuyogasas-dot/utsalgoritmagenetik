"""Loop evolusi Algoritma Genetika standar."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import operators as op
from .data_loader import DataJadwal
from .fitness import evaluasi, fitness


@dataclass
class GAConfig:
    """Parameter AG (default mengikuti PRD 4.4)."""

    pop_size: int = 100
    max_generasi: int = 500
    pc: float = 0.8
    pm: float = 0.05
    n_elite: int = 2
    k_turnamen: int = 3
    seleksi: str = "turnamen"      # 'roulette' | 'turnamen'
    crossover: str = "uniform"     # 'satu_titik' | 'uniform'
    paralel: bool = True           # aktifkan H5 (kelas A/B paralel)
    seed: int | None = None


@dataclass
class HasilGA:
    """Hasil satu run AG."""

    best_kromosom: np.ndarray
    best_fitness: float
    rincian: dict                       # pelanggaran H1-H5 individu terbaik
    mean_fitness_akhir: float
    history_best: list[float] = field(default_factory=list)
    history_mean: list[float] = field(default_factory=list)
    gen_valid_pertama: int | None = None   # generasi (1-based) saat fitness 1.0 pertama kali
    generasi_selesai: int = 0
    waktu: float = 0.0
    config: GAConfig = field(default_factory=GAConfig)

    @property
    def sukses(self) -> bool:
        return self.best_fitness >= 1.0


def _validasi_config(c: GAConfig) -> None:
    if c.pop_size < 2:
        raise ValueError("pop_size minimal 2")
    if not 0 <= c.n_elite <= c.pop_size:
        raise ValueError("n_elite harus 0..pop_size")
    if not (0 <= c.pc <= 1 and 0 <= c.pm <= 1):
        raise ValueError("pc dan pm harus dalam [0,1]")
    if c.max_generasi < 1:
        raise ValueError("max_generasi minimal 1")


def jalankan_ga(data: DataJadwal, config: GAConfig | None = None) -> HasilGA:
    """Jalankan AG. Seluruh keacakan berasal dari satu ``Generator`` ber-seed ``config.seed``."""
    cfg = config or GAConfig()
    _validasi_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()

    pop = op.inisialisasi_populasi(rng, cfg.pop_size, data.n_gen, data.max_gen)
    hist_best: list[float] = []
    hist_mean: list[float] = []
    gen_valid: int | None = None
    best_krom = pop[0].copy()
    best_fit = -1.0
    generasi = 0

    for generasi in range(1, cfg.max_generasi + 1):
        fit = fitness(pop, data, cfg.paralel)
        i_best = int(np.argmax(fit))
        hist_best.append(float(fit[i_best]))
        hist_mean.append(float(fit.mean()))
        if fit[i_best] > best_fit:
            best_fit, best_krom = float(fit[i_best]), pop[i_best].copy()
        if best_fit >= 1.0:
            if gen_valid is None:
                gen_valid = generasi
            break

        elit = op.elitisme(pop, fit, cfg.n_elite)
        n_anak = cfg.pop_size - cfg.n_elite
        n_pasang = (n_anak + 1) // 2
        induk = op.seleksi(cfg.seleksi, rng, pop, fit, 2 * n_pasang, cfg.k_turnamen)
        anak = []
        for i in range(n_pasang):
            c1, c2 = op.crossover(cfg.crossover, rng, induk[2 * i], induk[2 * i + 1], cfg.pc)
            anak.append(op.mutasi(rng, c1, cfg.pm, data.max_gen))
            anak.append(op.mutasi(rng, c2, cfg.pm, data.max_gen))
        pop = np.vstack([elit, np.array(anak[:n_anak]).reshape(n_anak, data.n_gen)])

    _, rincian = evaluasi(best_krom, data, cfg.paralel)
    return HasilGA(
        best_kromosom=best_krom,
        best_fitness=best_fit,
        rincian={k: int(v) for k, v in rincian.items()},
        mean_fitness_akhir=hist_mean[-1],
        history_best=hist_best,
        history_mean=hist_mean,
        gen_valid_pertama=gen_valid,
        generasi_selesai=generasi,
        waktu=time.perf_counter() - t0,
        config=cfg,
    )
