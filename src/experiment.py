"""Eksperimen batch (seed 1-10), ringkasan statistik, dan perbandingan konfigurasi."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .data_loader import DataJadwal
from .ga import GAConfig, HasilGA, jalankan_ga
from .verifier import verifikasi


def jalankan_batch(
    data: DataJadwal,
    config: GAConfig | None = None,
    seeds: range | list[int] = range(1, 11),
) -> tuple[pd.DataFrame, list[HasilGA]]:
    """Jalankan AG untuk setiap seed. Kembalikan DataFrame metrik dan daftar hasil lengkap."""
    dasar = config or GAConfig()
    baris, hasil_list = [], []
    for seed in seeds:
        h = jalankan_ga(data, replace(dasar, seed=seed))
        v = verifikasi(h.best_kromosom, data, dasar.paralel)
        hasil_list.append(h)
        baris.append(
            {
                "seed": seed,
                "fitness_terbaik": round(h.best_fitness, 6),
                "fitness_rata_akhir": round(h.mean_fitness_akhir, 6),
                "gen_valid_pertama": h.gen_valid_pertama if h.gen_valid_pertama is not None else "tidak ditemukan",
                "generasi_selesai": h.generasi_selesai,
                "pelanggaran_akhir": sum(h.rincian.values()),
                "terverifikasi_valid": v["valid"],
                "waktu_detik": round(h.waktu, 3),
            }
        )
    return pd.DataFrame(baris), hasil_list


def ringkasan(df: pd.DataFrame) -> dict:
    """Statistik ringkas: rata-rata/min/max fitness & generasi valid, jumlah run sukses, waktu."""
    gen = pd.to_numeric(df["gen_valid_pertama"], errors="coerce")
    return {
        "jumlah_run": len(df),
        "run_sukses": int((df["fitness_terbaik"] >= 1.0).sum()),
        "fitness_terbaik_rata2": float(df["fitness_terbaik"].mean()),
        "fitness_terbaik_min": float(df["fitness_terbaik"].min()),
        "fitness_terbaik_max": float(df["fitness_terbaik"].max()),
        "gen_valid_rata2": float(gen.mean()) if gen.notna().any() else None,
        "gen_valid_min": int(gen.min()) if gen.notna().any() else None,
        "gen_valid_max": int(gen.max()) if gen.notna().any() else None,
        "waktu_rata2_detik": float(df["waktu_detik"].mean()),
    }


def simpan_hasil(df: pd.DataFrame, path: str | Path = "output/hasil_10_run.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def cetak_ringkasan(df: pd.DataFrame, judul: str = "Hasil Batch") -> None:
    r = ringkasan(df)
    print(f"\n=== {judul} ===")
    print(df.to_string(index=False))
    print(f"\nRun sukses (fitness=1.0): {r['run_sukses']}/{r['jumlah_run']}")
    print(
        f"Fitness terbaik  rata2={r['fitness_terbaik_rata2']:.4f} "
        f"min={r['fitness_terbaik_min']:.4f} max={r['fitness_terbaik_max']:.4f}"
    )
    if r["gen_valid_rata2"] is not None:
        print(
            f"Generasi valid   rata2={r['gen_valid_rata2']:.1f} "
            f"min={r['gen_valid_min']} max={r['gen_valid_max']}"
        )
    print(f"Waktu rata-rata  {r['waktu_rata2_detik']:.3f} detik/run")


def bandingkan_konfigurasi(
    data: DataJadwal,
    konfigurasi: dict[str, GAConfig],
    seeds: range | list[int] = range(1, 11),
) -> pd.DataFrame:
    """Bandingkan beberapa konfigurasi pada seed yang sama; satu baris ringkasan per konfigurasi."""
    baris = []
    for nama, cfg in konfigurasi.items():
        df, _ = jalankan_batch(data, cfg, seeds)
        r = ringkasan(df)
        baris.append(
            {
                "konfigurasi": nama,
                "run_sukses": f"{r['run_sukses']}/{r['jumlah_run']}",
                "fitness_terbaik_rata2": round(r["fitness_terbaik_rata2"], 4),
                "gen_valid_rata2": round(r["gen_valid_rata2"], 1) if r["gen_valid_rata2"] is not None else None,
                "gen_valid_min": r["gen_valid_min"],
                "gen_valid_max": r["gen_valid_max"],
                "waktu_rata2_detik": round(r["waktu_rata2_detik"], 3),
            }
        )
    return pd.DataFrame(baris)


def konfigurasi_operator(dasar: GAConfig | None = None) -> dict[str, GAConfig]:
    """Empat kombinasi seleksi x crossover (T033)."""
    d = dasar or GAConfig()
    return {
        f"{s}+{c}": replace(d, seleksi=s, crossover=c)
        for s in ("roulette", "turnamen")
        for c in ("satu_titik", "uniform")
    }


def konfigurasi_paralel(dasar: GAConfig | None = None) -> dict[str, GAConfig]:
    """Paralel (H5) aktif vs nonaktif (T034, opsional)."""
    d = dasar or GAConfig()
    return {"paralel=True": replace(d, paralel=True), "paralel=False": replace(d, paralel=False)}
