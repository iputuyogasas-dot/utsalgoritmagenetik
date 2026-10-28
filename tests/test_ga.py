from dataclasses import replace

import numpy as np
import pytest

from src.export import ekspor_csv, ekspor_excel, jadwal_panjang, tabel_hari_slot
from src.fitness import fitness
from src.ga import GAConfig, jalankan_ga
from src.verifier import verifikasi

KECIL = GAConfig(pop_size=30, max_generasi=40, seed=7)


def test_elitisme_fitness_terbaik_tidak_turun(data):
    # Konfigurasi sulit agar tidak cepat berhenti; pm besar membuat elitisme krusial
    h = jalankan_ga(data, replace(KECIL, max_generasi=60, pm=0.3, pop_size=20))
    best = h.history_best
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))


@pytest.mark.parametrize("seleksi", ["roulette", "turnamen"])
@pytest.mark.parametrize("crossover", ["satu_titik", "uniform"])
def test_reproducible_dengan_seed_sama(data, seleksi, crossover):
    cfg = replace(KECIL, seleksi=seleksi, crossover=crossover)
    a, b = jalankan_ga(data, cfg), jalankan_ga(data, cfg)
    assert np.array_equal(a.best_kromosom, b.best_kromosom)
    assert a.history_best == b.history_best and a.history_mean == b.history_mean


def test_seed_berbeda_menghasilkan_jalur_berbeda(data):
    a = jalankan_ga(data, replace(KECIL, seed=1))
    b = jalankan_ga(data, replace(KECIL, seed=2))
    assert a.history_mean != b.history_mean


def test_ukuran_riwayat_dan_hasil(data):
    h = jalankan_ga(data, replace(KECIL, max_generasi=5, pm=0.0, pc=0.0))
    assert len(h.history_best) == len(h.history_mean) == h.generasi_selesai <= 5
    assert h.best_kromosom.shape == (data.n_gen,)
    assert fitness(h.best_kromosom, data) == pytest.approx(h.best_fitness)


def test_berhenti_saat_valid_dan_lolos_verifikator(data):
    h = jalankan_ga(data, GAConfig(seed=1))
    assert h.sukses and h.gen_valid_pertama == h.generasi_selesai < 500
    assert verifikasi(h.best_kromosom, data)["valid"]


def test_ga_tanpa_h5_lebih_mudah(data):
    h = jalankan_ga(data, GAConfig(seed=1, paralel=False))
    assert h.sukses


@pytest.mark.parametrize(
    "kw", [dict(pop_size=1), dict(n_elite=-1), dict(pc=1.5), dict(pm=-0.1), dict(max_generasi=0)]
)
def test_config_tidak_valid(data, kw):
    with pytest.raises(ValueError):
        jalankan_ga(data, replace(KECIL, **kw))


# ------------------------------------------------------------------------ ekspor
@pytest.fixture(scope="module")
def hasil_valid(data):
    return jalankan_ga(data, GAConfig(seed=1))


def test_jadwal_panjang_24_baris(data, hasil_valid):
    df = jadwal_panjang(hasil_valid.best_kromosom, data)
    assert len(df) == 24
    assert {"matkul", "kelas", "dosen", "hari", "jam", "ruang", "kelompok"} <= set(df.columns)
    assert df["jam"].str.match(r"\d\d\.\d\d–\d\d\.\d\d").all()


def test_tabel_hari_slot_bentuk_dan_isi(data, hasil_valid):
    t = tabel_hari_slot(hasil_valid.best_kromosom, data)
    assert t.shape == (3, 5)
    assert list(t.columns) == ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]
    isi = [c for c in t.to_numpy().ravel() if c != "-"]
    assert sum(c.count("|") // 3 for c in isi) == 24  # tiap kelas satu entri dengan 3 pemisah


def test_ekspor_file(data, hasil_valid, tmp_path):
    import pandas as pd

    c = ekspor_csv(hasil_valid.best_kromosom, data, tmp_path / "j.csv")
    x = ekspor_excel(hasil_valid.best_kromosom, data, tmp_path / "j.xlsx")
    assert len(pd.read_csv(c)) == 24
    sheets = pd.ExcelFile(x).sheet_names
    assert sum(s.startswith("Kelompok") for s in sheets) == 8
    assert sum(s.startswith("Dosen") for s in sheets) == 13
    assert len(set(sheets)) == len(sheets)
