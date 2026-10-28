"""Pengecekan kelayakan beban jadwal (PRD 3.4).

Jalankan: ``python -m src.feasibility``
"""
from __future__ import annotations

from collections import Counter

from .data_loader import DataJadwal, load_data


def cek_kelayakan(data: DataJadwal) -> dict:
    """Hitung beban per kelompok/dosen dan simpulkan apakah solusi tanpa bentrok layak."""
    beban_kelompok = Counter(s.kelompok for s in data.sesi)
    beban_dosen = Counter(s.dosen for s in data.sesi)
    n_slot = data.n_slot
    ruang_total = data.n_slot * data.n_ruang

    # Dengan paralel, tiap matkul memakai 1 slot -> kebutuhan slot = jumlah matkul
    n_matkul = len(data.pasangan)
    layak = (
        max(beban_kelompok.values()) <= n_slot
        and max(beban_dosen.values()) <= n_slot
        and n_matkul <= n_slot
        and data.n_gen <= ruang_total
        and int(data.jumlah_mhs.max()) <= int(data.kapasitas.max())
    )
    return {
        "beban_kelompok": dict(sorted(beban_kelompok.items())),
        "beban_dosen": dict(beban_dosen.most_common()),
        "n_sesi": data.n_gen,
        "n_matkul_paralel": n_matkul,
        "n_slot": n_slot,
        "kombinasi_slot_ruang": ruang_total,
        "layak": layak,
    }


def cetak_kelayakan(data: DataJadwal) -> None:
    h = cek_kelayakan(data)
    print("=== Cek Kelayakan Beban ===")
    print("Beban per kelompok mahasiswa (sesi/minggu):")
    for k, v in h["beban_kelompok"].items():
        print(f"  {k}: {v} (<= {h['n_slot']} slot)")
    dosen_top, beban_top = next(iter(h["beban_dosen"].items()))
    print(f"Dosen terpadat: {dosen_top} = {beban_top} kelas (<= {h['n_slot']} slot)")
    print(f"Matkul paralel memakai {h['n_matkul_paralel']} slot dari {h['n_slot']} tersedia")
    print(f"Total sesi {h['n_sesi']} <= {h['kombinasi_slot_ruang']} kombinasi slot-ruang")
    print("Kesimpulan:", "solusi tanpa bentrok LAYAK ada" if h["layak"] else "TIDAK layak")


if __name__ == "__main__":
    cetak_kelayakan(load_data())
