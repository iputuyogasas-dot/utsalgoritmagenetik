"""Grafik konvergensi fitness."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tanpa GUI; aman dijalankan di server/CLI
import matplotlib.pyplot as plt

from .ga import HasilGA


def plot_konvergensi(hasil: HasilGA, path: str | Path = "output/konvergensi.png", judul: str | None = None) -> Path:
    """Simpan grafik fitness terbaik dan rata-rata per generasi ke ``path`` (PNG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gen = range(1, len(hasil.history_best) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gen, hasil.history_best, label="Fitness terbaik", color="#1f77b4", linewidth=2)
    ax.plot(gen, hasil.history_mean, label="Fitness rata-rata", color="#ff7f0e", linewidth=1.5, alpha=0.85)
    if hasil.gen_valid_pertama is not None:
        ax.axvline(hasil.gen_valid_pertama, color="green", linestyle="--", alpha=0.6,
                   label=f"Solusi valid (gen {hasil.gen_valid_pertama})")
    c = hasil.config
    ax.set_title(judul or f"Konvergensi AG (seed={c.seed}, {c.seleksi} + {c.crossover})")
    ax.set_xlabel("Generasi")
    ax.set_ylabel("Fitness")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
