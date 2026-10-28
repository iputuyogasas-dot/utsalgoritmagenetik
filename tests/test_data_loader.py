import pandas as pd
import pytest

from src.data_loader import (
    DATA_DIR, bangun_sesi, load_data, load_matkul, load_pref_dosen, load_ruang, load_slot, validasi_data,
)


def test_data_asli_termuat(data):
    assert data.n_gen == 24
    assert data.n_ruang == 5 and data.n_slot == 15 and data.max_gen == 74
    assert len(data.pasangan) == 12
    assert len({s.kelompok for s in data.sesi}) == 8
    assert data.slot.loc[0, "jam_mulai"] == "07.30"  # tidak boleh menjadi 7.3


def test_pasangan_paralel_berdasarkan_kode(data):
    for a, b in data.pasangan:
        sa, sb = data.sesi[a], data.sesi[b]
        assert sa.kode == sb.kode and (sa.kelas, sb.kelas) == ("A", "B")
        assert sa.dosen != sb.dosen


def test_pref_dosen_terbaca():
    pref = load_pref_dosen()
    assert 2 in pref["Purmasari, S.T., M.T."]
    assert load_pref_dosen(DATA_DIR / "tidak_ada.csv") == {}


def test_semua_slot_dalam_jam_kerja(data):
    from src.data_loader import JAM_KERJA, _menit

    assert JAM_KERJA == ("07.00", "17.00")
    assert (data.slot["jam_mulai"].map(_menit) >= _menit("07.00")).all()
    assert (data.slot["jam_selesai"].map(_menit) <= _menit("17.00")).all()
    assert data.slot.groupby("hari").size().eq(3).all()  # 3 sesi per hari


def test_slot_di_luar_jam_kerja_ditolak():
    slot = load_slot()
    slot.loc[len(slot)] = [15, "Senin", 4, "15.30", "18.00"]  # berakhir setelah 17.00
    with pytest.raises(ValueError, match="di luar jam kerja"):
        validasi_data(load_matkul(), load_ruang(), slot)
    slot.loc[len(slot) - 1] = [15, "Senin", 0, "06.30", "09.00"]  # mulai sebelum 07.00
    with pytest.raises(ValueError, match="di luar jam kerja"):
        validasi_data(load_matkul(), load_ruang(), slot)


def test_pref_dosen_di_luar_rentang_ditolak(tmp_path):
    p = tmp_path / "pref.csv"
    p.write_text('dosen,slot_tidak_diinginkan\n"Purmasari, S.T., M.T.",2;24\n', encoding="utf-8")
    with pytest.raises(ValueError, match="di luar rentang"):
        load_data(pref_path=p)


def _ubah_matkul(fungsi):
    df = load_matkul()
    fungsi(df)
    return df


def test_file_tidak_ada():
    with pytest.raises(FileNotFoundError):
        load_matkul(DATA_DIR / "tidak_ada.csv")


def test_kolom_hilang(tmp_path):
    p = tmp_path / "matkul.csv"
    pd.read_csv(DATA_DIR / "matkul.csv").drop(columns=["dosen"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="kolom wajib hilang"):
        load_matkul(p)


def test_nilai_kosong(tmp_path):
    p = tmp_path / "matkul.csv"
    df = pd.read_csv(DATA_DIR / "matkul.csv")
    df.loc[2, "dosen"] = None
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="nilai kosong"):
        load_matkul(p)


def test_kode_tidak_berpasangan():
    df = _ubah_matkul(lambda d: d.__setitem__("kode", d["kode"].where(d["id"] != 2, "IF999")))
    with pytest.raises(ValueError, match="muncul 1x"):
        validasi_data(df, load_ruang(), load_slot())


def test_dosen_a_b_sama():
    def f(d):
        d.loc[d["id"] == 2, "dosen"] = d.loc[d["id"] == 1, "dosen"].iloc[0]
    with pytest.raises(ValueError, match="dosen kelas A dan B sama"):
        validasi_data(_ubah_matkul(f), load_ruang(), load_slot())


def test_kapasitas_tidak_cukup():
    def f(d):
        d.loc[d["id"] == 1, "jumlah_mhs"] = 99
    with pytest.raises(ValueError, match="kapasitas"):
        validasi_data(_ubah_matkul(f), load_ruang(), load_slot())


def test_sks_bukan_tiga():
    def f(d):
        d.loc[d["id"] == 1, "sks"] = 2
    with pytest.raises(ValueError, match="sks"):
        validasi_data(_ubah_matkul(f), load_ruang(), load_slot())


def test_id_duplikat():
    def f(d):
        d.loc[d["id"] == 2, "id"] = 1
    with pytest.raises(ValueError, match="id"):
        validasi_data(_ubah_matkul(f), load_ruang(), load_slot())


def test_bangun_sesi_urut_id():
    sesi, pasangan = bangun_sesi(load_matkul())
    assert [s.id for s in sesi] == list(range(1, 25))
    assert len(pasangan) == 12
