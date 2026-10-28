# Penjadwalan Kuliah dengan Algoritma Genetika (UTS)

Program Python yang menyusun jadwal kuliah **tanpa bentrok** (dosen, ruang, kelompok mahasiswa,
kapasitas, dan kelas paralel A/B) memakai Algoritma Genetika standar. Operator AG ditulis sendiri
(**tanpa DEAP/PyGAD**). PRD lengkap: [`docs/prd (2).md`](docs/prd%20(2).md).

## Persyaratan
- Python 3.10+
- `pip install -r requirements.txt` (NumPy, Pandas, Matplotlib, openpyxl, pytest)

## Cara menjalankan

```bash
# 1. Satu run (default: pop 100, 500 generasi, pc 0.8, pm 0.05, elite 2, turnamen k=3 + crossover uniform)
python main.py --seed 1

# 2. Ganti parameter / operator
python main.py --seed 3 --pop 150 --gen 800 --pc 0.9 --pm 0.03 --elite 2 \
               --selection roulette --crossover onepoint

# 3. Eksperimen batch 10 run (seed 1-10) + jadwal terbaik + grafik
python main.py --batch

# 4. Batch + perbandingan 4 kombinasi operator dan paralel on/off
python main.py --batch --bandingkan

# 5. Unit test
python -m pytest -q

# 6. Cek kelayakan beban data
python -m src.feasibility
```

| Opsi | Keterangan |
| --- | --- |
| `--seed` | Seed random (hasil reproducible), default 1 |
| `--pop`, `--gen` | Ukuran populasi, maksimum generasi |
| `--pc`, `--pm` | Peluang crossover, peluang mutasi per gen |
| `--elite`, `--k` | Jumlah elite, ukuran turnamen |
| `--selection` | `roulette` atau `tournament` |
| `--crossover` | `onepoint` atau `uniform` |
| `--no-paralel` | Nonaktifkan kendala H5 (kelas A/B paralel) |
| `--batch`, `--bandingkan` | Eksperimen 10 run; tambah perbandingan konfigurasi |
| `--output` | Folder keluaran, default `output` |

## Keluaran (folder `output/`)

| File | Isi |
| --- | --- |
| `jadwal_terbaik.csv` | Daftar panjang: matkul, kelas, dosen, hari, jam, ruang, kelompok |
| `jadwal_terbaik.xlsx` | Sheet jadwal mingguan (hari x slot), daftar panjang, per kelompok mahasiswa (8), dan per dosen (13) |
| `konvergensi.png` | Grafik fitness terbaik & rata-rata per generasi |
| `hasil_10_run.csv` | Metrik 10 run (mode `--batch`) |
| `perbandingan_operator.csv`, `perbandingan_paralel.csv` | Mode `--batch --bandingkan` |

## Struktur repositori

```
├── main.py              # CLI
├── data/                # matkul.csv, ruang.csv, slot.csv, pref_dosen.csv
├── src/
│   ├── data_loader.py   # baca & validasi data, bangun sesi + pasangan paralel
│   ├── chromosome.py    # encode/decode gen (slot, ruang)
│   ├── fitness.py       # H1-H5, fitness, soft constraint S1-S2 (informasi)
│   ├── verifier.py      # verifikator independen
│   ├── operators.py     # inisialisasi, seleksi, crossover, mutasi, elitisme
│   ├── ga.py            # loop evolusi
│   ├── experiment.py    # batch 10 run + statistik + perbandingan
│   ├── export.py        # tabel, CSV, Excel
│   ├── plot.py          # grafik konvergensi
│   └── feasibility.py   # cek kelayakan beban
├── tests/               # pytest
├── docs/                # PRD, tasks.json, draf laporan, slide
└── output/
```

## Ringkasan model
- **Kromosom:** array integer panjang 24 (satu gen per kelas); gen `g in [0,124]`, `slot = g // 5`, `ruang = g % 5`.
- **Fitness:** `1 / (1 + H1 + H2 + H3 + H4 + H5)`; bernilai 1,0 jika jadwal valid.
  H1 dosen, H2 ruang, H3 kelompok mahasiswa, H4 kapasitas, H5 kelas A/B harus satu slot.
- **Data:** `jumlah_mhs`, kapasitas ruang, penugasan dosen per kelas, dan preferensi dosen adalah
  **data dummy** (PRD 2.4) sampai ada data asli dari prodi; ganti langsung pada berkas CSV di `data/`.

## Hasil (default, seed 1-10)
Seluruh 10 run menemukan jadwal valid (rata-rata generasi ke-55,2; rentang 38-75; sekitar 0,1 detik/run),
dan semuanya lolos verifikator independen. Detail di [`docs/laporan_uts.md`](docs/laporan_uts.md).
