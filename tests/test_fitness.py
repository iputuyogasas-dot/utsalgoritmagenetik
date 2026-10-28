import numpy as np
import pytest

from src.chromosome import MAX_GEN, decode, encode, valid
from src.fitness import (
    evaluasi, fitness, h1_dosen, h2_ruang, h3_kelompok, h4_kapasitas, h5_paralel,
    pelanggaran, soft_constraint,
)
from src.verifier import verifikasi

from .conftest import taruh


# ------------------------------------------------------------------ kromosom
def test_encode_decode_bolak_balik():
    for s in range(15):
        for r in range(5):
            assert decode(encode(s, r)) == (s, r)


def test_valid_kromosom():
    assert valid(np.zeros(24, dtype=int))
    assert not valid(np.full(24, MAX_GEN + 1))
    assert not valid(np.zeros(23, dtype=int))


# ------------------------------------------------------------------- fitness
def test_jadwal_valid_fitness_satu(data, jadwal_valid):
    f, r = evaluasi(jadwal_valid, data)
    assert f == 1.0
    assert sum(r.values()) == 0


def test_h1_dosen_bentrok(data, jadwal_valid):
    # Purmasari mengampu idx 0 (Sem 1-A) dan idx 12 (Sem 3-A): satukan di slot 0 (ruang berbeda)
    k = jadwal_valid.copy()
    taruh(k, data, 12, 0, 2)
    taruh(k, data, 13, 0, 1)
    r = pelanggaran(k, data)
    assert r == {"H1": 1, "H2": 0, "H3": 0, "H4": 0, "H5": 0}
    assert fitness(k, data) == pytest.approx(1 / 2)


def test_h2_ruang_bentrok(data, jadwal_valid):
    # Sem 7 dipindah ke slot IMK (6); idx 22 satu ruang dengan idx 12 (FT-5)
    k = jadwal_valid.copy()
    taruh(k, data, 22, 6, 4)
    taruh(k, data, 23, 6, 2)
    r = pelanggaran(k, data)
    assert r == {"H1": 0, "H2": 1, "H3": 0, "H4": 0, "H5": 0}
    assert fitness(k, data) == pytest.approx(1 / 2)


def test_h3_kelompok_bentrok(data, jadwal_valid):
    # Logika Diskrit (idx 2,3) dipindah ke slot 0 bersama Aljabar: Sem 1-A dan Sem 1-B masing-masing bentrok
    k = jadwal_valid.copy()
    taruh(k, data, 2, 0, 2)
    taruh(k, data, 3, 0, 1)
    r = pelanggaran(k, data)
    assert r == {"H1": 0, "H2": 0, "H3": 2, "H4": 0, "H5": 0}
    assert fitness(k, data) == pytest.approx(1 / 3)


def test_h4_kapasitas(data, jadwal_valid):
    # Algoritma & Pemrograman A (40 mhs) di FT-1 (kapasitas 30)
    k = jadwal_valid.copy()
    taruh(k, data, 6, 3, 0)
    assert h4_kapasitas(k, data) == 1
    assert fitness(k, data) == pytest.approx(1 / 2)


def test_h5_paralel_beda_slot(data, jadwal_valid):
    k = jadwal_valid.copy()
    taruh(k, data, 7, 14, 3)  # kelas B pindah ke slot kosong lain
    r = pelanggaran(k, data)
    assert r == {"H1": 0, "H2": 0, "H3": 0, "H4": 0, "H5": 1}
    assert fitness(k, data) == pytest.approx(1 / 2)
    assert h5_paralel(k, data, paralel=False) == 0  # kendala dapat dimatikan


def test_fitness_populasi_sesuai_per_individu(data, jadwal_valid):
    rng = np.random.default_rng(0)
    pop = rng.integers(0, MAX_GEN + 1, size=(20, data.n_gen))
    pop[0] = jadwal_valid
    f = fitness(pop, data)
    assert f.shape == (20,)
    assert f[0] == 1.0
    for i in range(20):
        assert f[i] == pytest.approx(fitness(pop[i], data))


def test_fitness_turun_sesuai_pelanggaran(data, jadwal_valid):
    k = jadwal_valid.copy()
    taruh(k, data, 6, 3, 0)    # H4 = 1
    taruh(k, data, 7, 14, 3)   # H5 = 1
    assert fitness(k, data) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- soft constraint
def test_soft_constraint_tidak_mengubah_fitness(data, jadwal_valid):
    sebelum = fitness(jadwal_valid, data)
    info = soft_constraint(jadwal_valid, data)
    assert set(info) == {"S1_preferensi_dosen", "S2_jeda_kelompok"}
    assert fitness(jadwal_valid, data) == sebelum


def test_s2_jeda_dihitung(data, jadwal_valid):
    from src.fitness import s2_jeda_kelompok

    assert s2_jeda_kelompok(jadwal_valid, data) == 0  # tiap kelompok menempati sesi berurutan
    # Pindahkan Logika Diskrit (idx 2,3; Senin sesi 2) ke Jumat sesi 3: Senin Sem 1
    # tinggal sesi 1 dan 3 -> jeda 1 untuk Sem 1-A dan Sem 1-B
    k = jadwal_valid.copy()
    taruh(k, data, 2, 14, 4)
    taruh(k, data, 3, 14, 3)
    assert s2_jeda_kelompok(k, data) == 2


def test_s1_preferensi_dosen(data, jadwal_valid):
    from src.fitness import s1_preferensi_dosen

    # Pada jadwal valid ada 3 pelanggaran preferensi (data dummy): Christian Novan di slot 2,
    # Licantik di slot 3, dan Purmasari (Analisis Desain) di slot 8
    assert s1_preferensi_dosen(jadwal_valid, data) == 3
    k = jadwal_valid.copy()
    taruh(k, data, 0, 2, 2)  # Purmasari (tidak ingin slot 2) dipindah ke slot 2
    assert s1_preferensi_dosen(k, data) == 4



# ------------------------------------------------------------------ verifikator
def test_verifikator_jadwal_valid(data, jadwal_valid):
    v = verifikasi(jadwal_valid, data)
    assert v["valid"] and sum(v["jumlah"].values()) == 0 and v["detail"] == []


@pytest.mark.parametrize("paralel", [True, False])
def test_verifikator_konsisten_dengan_fitness(data, paralel):
    rng = np.random.default_rng(42)
    for _ in range(1000):
        k = rng.integers(0, MAX_GEN + 1, size=data.n_gen)
        v = verifikasi(k, data, paralel)["jumlah"]
        r = pelanggaran(k, data, paralel)
        assert v == {key: int(val) for key, val in r.items()}


def test_verifikator_mendeteksi_pelanggaran(data, jadwal_valid):
    k = jadwal_valid.copy()
    taruh(k, data, 7, 14, 3)
    v = verifikasi(k, data)
    assert not v["valid"] and v["jumlah"]["H5"] == 1
