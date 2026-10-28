"""Ekspor jadwal: tabel hari x slot, daftar panjang CSV, dan Excel multi-sheet."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import DataJadwal


def jadwal_panjang(kromosom: np.ndarray, data: DataJadwal) -> pd.DataFrame:
    """Daftar panjang: matkul, kelas, dosen, hari, jam, ruang, kelompok (diurut waktu)."""
    baris = []
    for s in data.sesi:
        g = int(kromosom[s.idx])
        sl, rg = g // data.n_ruang, g % data.n_ruang
        info = data.slot.iloc[sl]
        baris.append(
            {
                "matkul": s.matkul,
                "kelas": s.kelas,
                "dosen": s.dosen,
                "hari": info["hari"],
                "jam": f"{info['jam_mulai']}–{info['jam_selesai']}",
                "ruang": data.ruang.iloc[rg]["nama_ruang"],
                "kelompok": s.kelompok,
                "_slot": sl,
            }
        )
    df = pd.DataFrame(baris).sort_values(["_slot", "ruang", "kelompok"]).reset_index(drop=True)
    return df


def _tanpa_slot(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["_slot"])


def tabel_hari_slot(kromosom: np.ndarray, data: DataJadwal) -> pd.DataFrame:
    """Tabel jadwal mingguan: baris = jam sesi, kolom = hari.

    Tiap sel berisi ``Matkul (Kelas) | Dosen | Ruang | Kelompok`` (beberapa kelas
    pada slot sama dipisah baris baru). Sel kosong berisi ``"-"``.
    """
    hari = list(dict.fromkeys(data.slot["hari"]))
    jam = list(dict.fromkeys(f"{r.jam_mulai}–{r.jam_selesai}" for r in data.slot.itertuples()))
    tabel = pd.DataFrame("-", index=jam, columns=hari)
    tabel.index.name = "Slot"
    isi: dict[tuple[str, str], list[str]] = {}
    for _, r in jadwal_panjang(kromosom, data).iterrows():
        isi.setdefault((r["jam"], r["hari"]), []).append(
            f"{r['matkul']} ({r['kelas']}) | {r['dosen']} | {r['ruang']} | {r['kelompok']}"
        )
    for (j, h), teks in isi.items():
        tabel.loc[j, h] = "\n".join(teks)
    return tabel


def tabel_ringkas(kromosom: np.ndarray, data: DataJadwal) -> pd.DataFrame:
    """Versi ringkas tabel hari x slot untuk terminal: sel berisi ``kode-kelas@ruang``."""
    hari = list(dict.fromkeys(data.slot["hari"]))
    jam = list(dict.fromkeys(f"{r.jam_mulai}–{r.jam_selesai}" for r in data.slot.itertuples()))
    tabel = pd.DataFrame("-", index=jam, columns=hari)
    tabel.index.name = "Slot"
    isi: dict[tuple[str, str], list[str]] = {}
    kode = {(s.matkul, s.kelas): s.kode for s in data.sesi}
    for _, r in jadwal_panjang(kromosom, data).iterrows():
        isi.setdefault((r["jam"], r["hari"]), []).append(f"{kode[(r['matkul'], r['kelas'])]}-{r['kelas']}@{r['ruang']}")
    for (j, h), teks in isi.items():
        tabel.loc[j, h] = " ".join(teks)
    return tabel


def ekspor_csv(kromosom: np.ndarray, data: DataJadwal, path: str | Path = "output/jadwal_terbaik.csv") -> Path:
    """Simpan daftar panjang ke CSV (UTF-8 dengan BOM agar terbuka benar di Excel)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _tanpa_slot(jadwal_panjang(kromosom, data)).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _nama_sheet(awal: str, dipakai: set[str]) -> str:
    """Nama sheet Excel valid (maks 31 karakter, tanpa karakter terlarang, unik)."""
    nama = re.sub(r"[\[\]:*?/\\]", "", awal).strip()[:31]
    dasar, i = nama, 2
    while nama in dipakai:
        sufiks = f"~{i}"
        nama = dasar[: 31 - len(sufiks)] + sufiks
        i += 1
    dipakai.add(nama)
    return nama


def ekspor_excel(kromosom: np.ndarray, data: DataJadwal, path: str | Path = "output/jadwal_terbaik.xlsx") -> Path:
    """Simpan Excel: sheet mingguan, daftar panjang, serta sheet per kelompok & per dosen."""
    from openpyxl.styles import Alignment

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panjang = _tanpa_slot(jadwal_panjang(kromosom, data))
    dipakai: set[str] = set()

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        tabel_hari_slot(kromosom, data).to_excel(xw, sheet_name=_nama_sheet("Jadwal Mingguan", dipakai))
        panjang.to_excel(xw, sheet_name=_nama_sheet("Daftar Jadwal", dipakai), index=False)
        for kel in sorted(panjang["kelompok"].unique()):
            panjang[panjang["kelompok"] == kel].to_excel(
                xw, sheet_name=_nama_sheet(f"Kelompok {kel}", dipakai), index=False
            )
        for dsn in sorted(panjang["dosen"].unique()):
            panjang[panjang["dosen"] == dsn].to_excel(
                xw, sheet_name=_nama_sheet(f"Dosen {dsn.split(',')[0]}", dipakai), index=False
            )
        for ws in xw.book.worksheets:
            for kolom in ws.columns:
                lebar = max(len(str(c.value).split("\n")[0]) if c.value is not None else 0 for c in kolom)
                ws.column_dimensions[kolom[0].column_letter].width = min(max(lebar + 2, 10), 60)
            for baris in ws.iter_rows():
                for c in baris:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    return path
