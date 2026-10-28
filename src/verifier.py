"""Verifikator independen untuk jadwal akhir.

Sengaja ditulis terpisah dari ``fitness.py``: tidak memakai array turunan
``DataJadwal`` ataupun fungsi fitness, melainkan mengelompokkan sesi per slot
dengan loop eksplisit dari data mentah (``data.sesi`` dan ``data.ruang``).
"""
from __future__ import annotations

from collections import defaultdict

from .data_loader import DataJadwal


def verifikasi(kromosom, data: DataJadwal, paralel: bool = True) -> dict:
    """Periksa jadwal; kembalikan ``{'valid': bool, 'jumlah': {H1..H5}, 'detail': [str,...]}``."""
    n_ruang = len(data.ruang)
    kapasitas = {int(r.id_ruang): int(r.kapasitas) for r in data.ruang.itertuples(index=False)}
    detail: list[str] = []
    jumlah = {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "H5": 0}

    slot_of: dict[int, int] = {}
    ruang_of: dict[int, int] = {}
    for s in data.sesi:
        g = int(kromosom[s.idx])
        slot_of[s.idx] = g // n_ruang
        ruang_of[s.idx] = g % n_ruang

    # Kelompokkan sesi per slot, lalu periksa tiap slot
    per_slot: dict[int, list] = defaultdict(list)
    for s in data.sesi:
        per_slot[slot_of[s.idx]].append(s)

    for sl, daftar in per_slot.items():
        for a in range(len(daftar)):
            for b in range(a + 1, len(daftar)):
                x, y = daftar[a], daftar[b]
                if x.dosen == y.dosen:
                    jumlah["H1"] += 1
                    detail.append(f"H1 slot {sl}: dosen {x.dosen} mengajar id {x.id} & {y.id}")
                if ruang_of[x.idx] == ruang_of[y.idx]:
                    jumlah["H2"] += 1
                    detail.append(f"H2 slot {sl}: ruang {ruang_of[x.idx]} dipakai id {x.id} & {y.id}")
                if x.kelompok == y.kelompok:
                    jumlah["H3"] += 1
                    detail.append(f"H3 slot {sl}: kelompok {x.kelompok} punya id {x.id} & {y.id}")

    for s in data.sesi:
        if kapasitas[ruang_of[s.idx]] < s.jumlah_mhs:
            jumlah["H4"] += 1
            detail.append(
                f"H4 id {s.id}: {s.jumlah_mhs} mhs > kapasitas ruang {kapasitas[ruang_of[s.idx]]}"
            )

    if paralel:
        per_kode: dict[str, list] = defaultdict(list)
        for s in data.sesi:
            per_kode[s.kode].append(s)
        for kode, kelas in per_kode.items():
            if len({slot_of[k.idx] for k in kelas}) > 1:
                jumlah["H5"] += 1
                detail.append(f"H5 kode {kode}: kelas A/B berada di slot berbeda")

    return {"valid": sum(jumlah.values()) == 0, "jumlah": jumlah, "detail": detail}
