"""CLI Penjadwalan Kuliah dengan Algoritma Genetika.

Contoh:
    python main.py --seed 1
    python main.py --seed 3 --pop 150 --gen 800 --selection roulette --crossover onepoint
    python main.py --batch                 # 10 run (seed 1-10) + ekspor + grafik
    python main.py --batch --bandingkan    # + perbandingan operator & paralel on/off
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src import experiment as exp
from src.data_loader import load_data
from src.export import ekspor_csv, ekspor_excel, tabel_ringkas
from src.fitness import soft_constraint
from src.ga import GAConfig, HasilGA, jalankan_ga
from src.plot import plot_konvergensi
from src.verifier import verifikasi

PILIHAN_SELEKSI = {"roulette": "roulette", "tournament": "turnamen", "turnamen": "turnamen"}
PILIHAN_CROSSOVER = {"onepoint": "satu_titik", "satu_titik": "satu_titik", "uniform": "uniform"}


def buat_parser() -> argparse.ArgumentParser:
    d = GAConfig()
    p = argparse.ArgumentParser(description="Penjadwalan kuliah dengan Algoritma Genetika (UTS)")
    p.add_argument("--seed", type=int, default=1, help="seed random (default 1)")
    p.add_argument("--pop", type=int, default=d.pop_size, help="ukuran populasi")
    p.add_argument("--gen", type=int, default=d.max_generasi, help="maksimum generasi")
    p.add_argument("--pc", type=float, default=d.pc, help="peluang crossover")
    p.add_argument("--pm", type=float, default=d.pm, help="peluang mutasi per gen")
    p.add_argument("--elite", type=int, default=d.n_elite, help="jumlah elite")
    p.add_argument("--k", type=int, default=d.k_turnamen, help="ukuran turnamen")
    p.add_argument("--selection", choices=sorted(PILIHAN_SELEKSI), default="tournament")
    p.add_argument("--crossover", choices=sorted(PILIHAN_CROSSOVER), default="uniform")
    p.add_argument("--no-paralel", action="store_true", help="nonaktifkan H5 (kelas A/B paralel)")
    p.add_argument("--batch", action="store_true", help="eksperimen batch seed 1-10")
    p.add_argument("--bandingkan", action="store_true", help="(dengan --batch) bandingkan operator & paralel")
    p.add_argument("--output", default="output", help="folder keluaran (default output)")
    return p


def cetak_hasil_run(h: HasilGA, data) -> None:
    c = h.config
    v = verifikasi(h.best_kromosom, data, c.paralel)
    print(f"\n=== Hasil Run (seed={c.seed}, {c.seleksi}+{c.crossover}, pop={c.pop_size}) ===")
    print(f"Generasi selesai : {h.generasi_selesai}/{c.max_generasi}")
    print(f"Generasi valid   : {h.gen_valid_pertama if h.gen_valid_pertama else 'tidak ditemukan'}")
    print(f"Fitness terbaik  : {h.best_fitness:.6f}")
    print(f"Pelanggaran      : {h.rincian}")
    print(f"Verifikator      : {'VALID (0 pelanggaran H1-H5)' if v['valid'] else 'TIDAK VALID'}")
    for d in v["detail"][:10]:
        print("   -", d)
    print(f"Soft constraint  : {soft_constraint(h.best_kromosom, data)}  (informasi saja)")
    print(f"Waktu            : {h.waktu:.3f} detik")


def simpan_output(h: HasilGA, data, folder: Path) -> None:
    """Cetak tabel jadwal lalu ekspor CSV, Excel, dan grafik konvergensi."""
    tabel = tabel_ringkas(h.best_kromosom, data)
    print("\n=== Jadwal Terbaik (hari x slot; sel = kode-kelas@ruang) ===")
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(tabel.to_string())
    print("(Detail lengkap tiap sel: lihat jadwal_terbaik.xlsx / jadwal_terbaik.csv)")
    f1 = ekspor_csv(h.best_kromosom, data, folder / "jadwal_terbaik.csv")
    f2 = ekspor_excel(h.best_kromosom, data, folder / "jadwal_terbaik.xlsx")
    f3 = plot_konvergensi(h, folder / "konvergensi.png")
    print(f"\nFile tersimpan: {f1}, {f2}, {f3}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a = buat_parser().parse_args(argv)
    cfg = GAConfig(
        pop_size=a.pop, max_generasi=a.gen, pc=a.pc, pm=a.pm, n_elite=a.elite, k_turnamen=a.k,
        seleksi=PILIHAN_SELEKSI[a.selection], crossover=PILIHAN_CROSSOVER[a.crossover],
        paralel=not a.no_paralel, seed=a.seed,
    )
    data = load_data()
    folder = Path(a.output)

    if not a.batch:
        h = jalankan_ga(data, cfg)
        cetak_hasil_run(h, data)
        simpan_output(h, data, folder)
        return 0 if h.sukses else 1

    df, hasil = exp.jalankan_batch(data, cfg, range(1, 11))
    exp.cetak_ringkasan(df, "Hasil 10 Run (seed 1-10)")
    print("File tersimpan:", exp.simpan_hasil(df, folder / "hasil_10_run.csv"))
    # Jadwal terbaik = fitness tertinggi, seri dipecah oleh generasi valid tercepat
    terbaik = min(hasil, key=lambda h: (-h.best_fitness, h.gen_valid_pertama or 10**9))
    cetak_hasil_run(terbaik, data)
    simpan_output(terbaik, data, folder)

    if a.bandingkan:
        for nama, konf in (("perbandingan_operator", exp.konfigurasi_operator(replace(cfg, seed=None))),
                           ("perbandingan_paralel", exp.konfigurasi_paralel(replace(cfg, seed=None)))):
            tbl = exp.bandingkan_konfigurasi(data, konf)
            print(f"\n=== {nama} ===")
            print(tbl.to_string(index=False))
            tbl.to_csv(folder / f"{nama}.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
