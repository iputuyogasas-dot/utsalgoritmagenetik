"""Pemuatan dan validasi data penjadwalan, serta pembangunan daftar sesi.

Satu *sesi* = satu kelas (satu baris di ``matkul.csv``). Kunci unik sesi adalah
``id`` (1..N); pasangan paralel kelas A/B ditentukan dari kolom ``kode``.
Indeks gen pada kromosom (0..N-1) mengikuti urutan baris ``matkul.csv`` yang
sudah diurutkan berdasarkan ``id``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KOLOM_MATKUL = ["id", "matkul", "kelas", "kode", "semester", "sks", "dosen", "jumlah_mhs"]
KOLOM_RUANG = ["id_ruang", "nama_ruang", "kapasitas"]
KOLOM_SLOT = ["id_slot", "hari", "sesi", "jam_mulai", "jam_selesai"]

# Jam kerja dosen di kampus: seluruh slot harus berada di dalam rentang ini (format "JJ.MM")
JAM_KERJA = ("07.00", "17.00")


def _menit(jam: str) -> int:
    """Ubah teks "JJ.MM" (atau "JJ:MM") menjadi menit sejak tengah malam."""
    j, m = str(jam).strip().replace(":", ".").split(".")
    return int(j) * 60 + int(m)


@dataclass(frozen=True)
class Sesi:
    """Satu sesi kuliah (satu kelas) yang menempati satu gen kromosom."""

    idx: int          # indeks gen (0..N-1)
    id: int           # id pada matkul.csv
    matkul: str
    kelas: str
    kode: str
    semester: int
    dosen: str
    kelompok: str     # mis. "Sem 1-A"
    jumlah_mhs: int


@dataclass
class DataJadwal:
    """Seluruh data masukan yang sudah divalidasi + struktur turunan untuk fitness."""

    sesi: list[Sesi]
    ruang: pd.DataFrame
    slot: pd.DataFrame
    pasangan: list[tuple[int, int]]            # pasangan paralel (idx A, idx B)
    pref_dosen: dict[str, set[int]] = field(default_factory=dict)

    # --- array turunan (diisi di __post_init__) ---
    dosen_id: np.ndarray = field(init=False)
    kelompok_id: np.ndarray = field(init=False)
    jumlah_mhs: np.ndarray = field(init=False)
    kapasitas: np.ndarray = field(init=False)
    slot_hari: np.ndarray = field(init=False)        # indeks hari tiap slot
    slot_sesi: np.ndarray = field(init=False)        # nomor sesi dalam hari tiap slot
    pairs_dosen: np.ndarray = field(init=False)      # (m,2) pasangan sesi satu dosen
    pairs_kelompok: np.ndarray = field(init=False)   # (m,2) pasangan sesi satu kelompok
    pairs_all: np.ndarray = field(init=False)        # (N(N-1)/2, 2) semua pasangan
    pair_a: np.ndarray = field(init=False)
    pair_b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.dosen_id = _faktorkan([s.dosen for s in self.sesi])
        self.kelompok_id = _faktorkan([s.kelompok for s in self.sesi])
        self.jumlah_mhs = np.array([s.jumlah_mhs for s in self.sesi], dtype=int)
        self.kapasitas = self.ruang["kapasitas"].to_numpy(dtype=int)
        self.slot_hari = _faktorkan(self.slot["hari"].tolist())
        self.slot_sesi = self.slot["sesi"].to_numpy(dtype=int)
        n = len(self.sesi)
        iu = np.triu_indices(n, k=1)
        self.pairs_all = np.column_stack(iu)
        sama_dosen = self.dosen_id[iu[0]] == self.dosen_id[iu[1]]
        sama_kel = self.kelompok_id[iu[0]] == self.kelompok_id[iu[1]]
        self.pairs_dosen = self.pairs_all[sama_dosen]
        self.pairs_kelompok = self.pairs_all[sama_kel]
        self.pair_a = np.array([p[0] for p in self.pasangan], dtype=int)
        self.pair_b = np.array([p[1] for p in self.pasangan], dtype=int)

    @property
    def n_gen(self) -> int:
        """Jumlah gen (= jumlah sesi)."""
        return len(self.sesi)

    @property
    def n_ruang(self) -> int:
        return len(self.ruang)

    @property
    def n_slot(self) -> int:
        return len(self.slot)

    @property
    def max_gen(self) -> int:
        """Nilai gen maksimum (inklusif): n_slot * n_ruang - 1."""
        return self.n_slot * self.n_ruang - 1


def _faktorkan(nilai: list[str]) -> np.ndarray:
    """Ubah daftar label menjadi id integer berurutan kemunculan."""
    peta: dict[str, int] = {}
    return np.array([peta.setdefault(v, len(peta)) for v in nilai], dtype=int)


def _baca_csv(path: Path, kolom_wajib: list[str], dtype: dict | None = None) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")
    df = pd.read_csv(path, dtype=dtype)
    hilang = [k for k in kolom_wajib if k not in df.columns]
    if hilang:
        raise ValueError(f"{Path(path).name}: kolom wajib hilang: {hilang}")
    if df[kolom_wajib].isnull().any().any():
        baris = df[df[kolom_wajib].isnull().any(axis=1)].index.tolist()
        raise ValueError(f"{Path(path).name}: ada nilai kosong pada baris (0-based) {baris}")
    return df


def load_matkul(path: Path | str | None = None) -> pd.DataFrame:
    """Baca ``matkul.csv`` dan cek kolom, tipe, serta nilai kosong."""
    df = _baca_csv(Path(path) if path else DATA_DIR / "matkul.csv", KOLOM_MATKUL)
    for k in ["id", "semester", "sks", "jumlah_mhs"]:
        try:
            df[k] = df[k].astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"matkul.csv: kolom '{k}' harus berupa bilangan bulat") from exc
    df["kelas"] = df["kelas"].astype(str).str.strip().str.upper()
    return df.sort_values("id").reset_index(drop=True)


def load_ruang(path: Path | str | None = None) -> pd.DataFrame:
    """Baca ``ruang.csv``; ``id_ruang`` harus 0..R-1 berurutan (sama dengan ``g % R``)."""
    df = _baca_csv(Path(path) if path else DATA_DIR / "ruang.csv", KOLOM_RUANG)
    df["id_ruang"] = df["id_ruang"].astype(int)
    df["kapasitas"] = df["kapasitas"].astype(int)
    df = df.sort_values("id_ruang").reset_index(drop=True)
    if df["id_ruang"].tolist() != list(range(len(df))):
        raise ValueError("ruang.csv: id_ruang harus 0..R-1 berurutan tanpa celah")
    return df


def load_slot(path: Path | str | None = None) -> pd.DataFrame:
    """Baca ``slot.csv``; ``id_slot`` harus 0..S-1 berurutan."""
    df = _baca_csv(
        Path(path) if path else DATA_DIR / "slot.csv", KOLOM_SLOT,
        dtype={"jam_mulai": str, "jam_selesai": str},
    )
    df["id_slot"] = df["id_slot"].astype(int)
    df["sesi"] = df["sesi"].astype(int)
    df = df.sort_values("id_slot").reset_index(drop=True)
    if df["id_slot"].tolist() != list(range(len(df))):
        raise ValueError("slot.csv: id_slot harus 0..S-1 berurutan tanpa celah")
    return df


def load_pref_dosen(path: Path | str | None = None) -> dict[str, set[int]]:
    """Baca ``pref_dosen.csv`` (opsional). Kembalikan ``{dosen: {id_slot, ...}}``.

    Kolom ``slot_tidak_diinginkan`` berisi id slot dipisah titik koma. File yang
    tidak ada menghasilkan dict kosong.
    """
    p = Path(path) if path else DATA_DIR / "pref_dosen.csv"
    if not p.exists():
        return {}
    df = _baca_csv(p, ["dosen", "slot_tidak_diinginkan"])
    hasil: dict[str, set[int]] = {}
    for _, r in df.iterrows():
        teks = str(r["slot_tidak_diinginkan"]).strip()
        hasil[r["dosen"]] = {int(x) for x in teks.split(";") if x.strip() != ""}
    return hasil


def validasi_data(matkul: pd.DataFrame, ruang: pd.DataFrame, slot: pd.DataFrame) -> None:
    """Validasi kualitas data (PRD 2.4). Melempar ``ValueError`` jika ada masalah.

    Pemeriksaan: id unik 1..N, sks = 3, tiap ``kode`` tepat 2 baris (kelas A dan B,
    semester & matkul sama), dosen pasangan A/B berbeda, dan tiap kelas muat di
    minimal satu ruang.
    """
    masalah: list[str] = []

    if matkul["id"].duplicated().any():
        masalah.append(f"id duplikat: {matkul.loc[matkul['id'].duplicated(), 'id'].tolist()}")
    elif matkul["id"].tolist() != list(range(1, len(matkul) + 1)):
        masalah.append("id harus berurutan 1..N")

    if (matkul["sks"] != 3).any():
        masalah.append(f"sks bukan 3 pada id: {matkul.loc[matkul['sks'] != 3, 'id'].tolist()}")

    for kode, grup in matkul.groupby("kode"):
        if len(grup) != 2:
            masalah.append(f"kode '{kode}' muncul {len(grup)}x (harus tepat 2: kelas A dan B)")
            continue
        if sorted(grup["kelas"]) != ["A", "B"]:
            masalah.append(f"kode '{kode}' harus berisi kelas A dan B, ditemukan {grup['kelas'].tolist()}")
        if grup["semester"].nunique() != 1 or grup["matkul"].nunique() != 1:
            masalah.append(f"kode '{kode}': semester/matkul kelas A dan B berbeda")
        if grup["dosen"].nunique() != 2:
            masalah.append(f"kode '{kode}': dosen kelas A dan B sama (harus berbeda)")

    kap_maks = int(ruang["kapasitas"].max())
    besar = matkul[matkul["jumlah_mhs"] > kap_maks]
    if not besar.empty:
        masalah.append(
            f"kapasitas ruang terbesar ({kap_maks}) tidak muat untuk id: {besar['id'].tolist()}"
        )

    # Seluruh slot harus berada dalam jam kerja kampus
    awal, akhir = _menit(JAM_KERJA[0]), _menit(JAM_KERJA[1])
    for r in slot.itertuples(index=False):
        if _menit(r.jam_mulai) < awal or _menit(r.jam_selesai) > akhir:
            masalah.append(
                f"slot {r.id_slot} ({r.hari} {r.jam_mulai}-{r.jam_selesai}) di luar jam kerja "
                f"{JAM_KERJA[0]}-{JAM_KERJA[1]}"
            )

    if masalah:
        raise ValueError("Validasi data gagal:\n- " + "\n- ".join(masalah))


def bangun_sesi(matkul: pd.DataFrame) -> tuple[list[Sesi], list[tuple[int, int]]]:
    """Bangun daftar sesi dan pasangan paralel (idx kelas A, idx kelas B)."""
    sesi = [
        Sesi(
            idx=i,
            id=int(r.id),
            matkul=r.matkul,
            kelas=r.kelas,
            kode=r.kode,
            semester=int(r.semester),
            dosen=r.dosen,
            kelompok=f"Sem {int(r.semester)}-{r.kelas}",
            jumlah_mhs=int(r.jumlah_mhs),
        )
        for i, r in enumerate(matkul.itertuples(index=False))
    ]
    per_kode: dict[str, dict[str, int]] = {}
    for s in sesi:
        per_kode.setdefault(s.kode, {})[s.kelas] = s.idx
    pasangan = [(d["A"], d["B"]) for d in per_kode.values()]
    return sesi, pasangan


def load_data(
    matkul_path: Path | str | None = None,
    ruang_path: Path | str | None = None,
    slot_path: Path | str | None = None,
    pref_path: Path | str | None = None,
) -> DataJadwal:
    """Muat, validasi, dan bangun seluruh data penjadwalan."""
    matkul = load_matkul(matkul_path)
    ruang = load_ruang(ruang_path)
    slot = load_slot(slot_path)
    validasi_data(matkul, ruang, slot)
    sesi, pasangan = bangun_sesi(matkul)
    pref = load_pref_dosen(pref_path)
    for dosen, slots in pref.items():
        di_luar = sorted(s for s in slots if not 0 <= s < len(slot))
        if di_luar:
            raise ValueError(f"pref_dosen.csv: slot {di_luar} untuk '{dosen}' di luar rentang 0..{len(slot) - 1}")
    return DataJadwal(sesi=sesi, ruang=ruang, slot=slot, pasangan=pasangan, pref_dosen=pref)
