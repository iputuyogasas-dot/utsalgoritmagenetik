import numpy as np
import pytest

from src import operators as op
from src.chromosome import MAX_GEN

N = 24


@pytest.fixture
def pop_fit():
    rng = np.random.default_rng(0)
    pop = rng.integers(0, MAX_GEN + 1, size=(30, N))
    fit = rng.random(30) + 0.01
    return pop, fit


def _valid(k):
    return k.shape == (N,) and k.min() >= 0 and k.max() <= MAX_GEN


# ------------------------------------------------------------------ inisialisasi
def test_inisialisasi_shape_rentang_dan_seed():
    a = op.inisialisasi_populasi(np.random.default_rng(1), 50, N, MAX_GEN)
    b = op.inisialisasi_populasi(np.random.default_rng(1), 50, N, MAX_GEN)
    assert a.shape == (50, N) and a.min() >= 0 and a.max() <= MAX_GEN
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------- seleksi
@pytest.mark.parametrize("metode", ["roulette", "turnamen"])
def test_seleksi_menghasilkan_anggota_populasi(pop_fit, metode):
    pop, fit = pop_fit
    rng = np.random.default_rng(3)
    dipilih = op.seleksi(metode, rng, pop, fit, 40, 3)
    assert dipilih.shape == (40, N)
    baris_pop = {tuple(r) for r in pop}
    assert all(tuple(r) in baris_pop for r in dipilih)


def test_seleksi_tidak_mengubah_populasi(pop_fit):
    pop, fit = pop_fit
    salinan = pop.copy()
    d = op.seleksi("turnamen", np.random.default_rng(0), pop, fit, 10, 3)
    d[:] = -1
    assert np.array_equal(pop, salinan)


def test_roulette_condong_ke_fitness_tinggi():
    pop = np.arange(2 * N).reshape(2, N) % (MAX_GEN + 1)
    fit = np.array([0.9, 0.1])
    dipilih = op.seleksi_roulette(np.random.default_rng(0), pop, fit, 2000)
    frac0 = np.mean(np.all(dipilih == pop[0], axis=1))
    assert 0.85 < frac0 < 0.95


def test_turnamen_k_besar_memilih_terbaik(pop_fit):
    pop, fit = pop_fit
    dipilih = op.seleksi_turnamen(np.random.default_rng(0), pop, fit, 200, k=200)
    terbaik = pop[np.argmax(fit)]
    assert np.mean(np.all(dipilih == terbaik, axis=1)) > 0.95


def test_seleksi_metode_tidak_dikenal(pop_fit):
    pop, fit = pop_fit
    with pytest.raises(ValueError):
        op.seleksi("acak", np.random.default_rng(0), pop, fit, 2)


# ---------------------------------------------------------------------- crossover
@pytest.mark.parametrize("metode", ["satu_titik", "uniform"])
@pytest.mark.parametrize("seed", range(20))
def test_crossover_panjang_rentang_induk_utuh(metode, seed):
    rng = np.random.default_rng(seed)
    p1 = rng.integers(0, MAX_GEN + 1, N)
    p2 = rng.integers(0, MAX_GEN + 1, N)
    p1c, p2c = p1.copy(), p2.copy()
    c1, c2 = op.crossover(metode, rng, p1, p2, 1.0)
    assert _valid(c1) and _valid(c2)
    assert np.array_equal(p1, p1c) and np.array_equal(p2, p2c)
    # tiap gen anak berasal dari salah satu induk pada posisi yang sama
    assert np.all((c1 == p1) | (c1 == p2)) and np.all((c2 == p1) | (c2 == p2))


@pytest.mark.parametrize("metode", ["satu_titik", "uniform"])
def test_crossover_pc_nol_menyalin_induk(metode):
    rng = np.random.default_rng(0)
    p1 = rng.integers(0, MAX_GEN + 1, N)
    p2 = rng.integers(0, MAX_GEN + 1, N)
    c1, c2 = op.crossover(metode, rng, p1, p2, 0.0)
    assert np.array_equal(c1, p1) and np.array_equal(c2, p2)


def test_crossover_satu_titik_berbentuk_awal_dan_ekor():
    rng = np.random.default_rng(5)
    p1 = np.zeros(N, dtype=int)
    p2 = np.ones(N, dtype=int)
    c1, _ = op.crossover_satu_titik(rng, p1, p2, 1.0)
    titik = int(np.argmax(c1))
    assert 1 <= titik < N and np.all(c1[:titik] == 0) and np.all(c1[titik:] == 1)


def test_crossover_metode_tidak_dikenal():
    with pytest.raises(ValueError):
        op.crossover("dua_titik", np.random.default_rng(0), np.zeros(N, int), np.zeros(N, int), 1.0)


# ------------------------------------------------------------------------ mutasi
def test_mutasi_pm_nol_tidak_berubah():
    k = np.random.default_rng(0).integers(0, MAX_GEN + 1, N)
    assert np.array_equal(op.mutasi(np.random.default_rng(1), k, 0.0, MAX_GEN), k)


def test_mutasi_pm_satu_mengubah_banyak_gen_dan_valid():
    k = np.full(N, 7)
    m = op.mutasi(np.random.default_rng(1), k, 1.0, MAX_GEN)
    assert _valid(m) and np.sum(m != 7) >= N // 2
    assert np.all(k == 7)  # kromosom asli utuh


def test_mutasi_laju_mendekati_pm():
    rng = np.random.default_rng(0)
    k = np.full(N, -1)  # nilai di luar rentang -> setiap perubahan terdeteksi
    jumlah = sum(int(np.sum(op.mutasi(rng, k, 0.1, MAX_GEN) != -1)) for _ in range(500))
    assert 0.07 < jumlah / (500 * N) < 0.13


# ---------------------------------------------------------------------- elitisme
def test_elitisme_memilih_terbaik(pop_fit):
    pop, fit = pop_fit
    elit = op.elitisme(pop, fit, 2)
    urut = np.argsort(-fit)[:2]
    assert np.array_equal(elit, pop[urut])
    elit[:] = 0
    assert not np.all(pop[urut] == 0)  # salinan, bukan view


def test_elitisme_nol():
    pop = np.zeros((5, N), dtype=int)
    assert op.elitisme(pop, np.ones(5), 0).shape == (0, N)
