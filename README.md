# Klasifikasi Penyakit Paru-Paru dengan PyTorch

Proyek ini membandingkan empat model *deep learning* untuk mengklasifikasikan
citra X-Ray dada ke dalam tiga kelas: `Lung_Opacity`, `Normal`, dan
`Viral Pneumonia`.

## Ringkasan

- **EnhancedResidualSE_CNN** — CNN kustom dengan residual connection dan
  *squeeze-and-excitation*.
- **EfficientNetV2-B0**, **ConvNeXt-Tiny**, dan **Xception** — model
  *transfer learning* dari pustaka `timm`, dilatih melalui tahap *classifier
  warmup* dan *fine-tuning*.
- Evaluasi menggunakan pembagian data terstratifikasi 70% train, 15%
  validation, dan 15% test serta *test-time augmentation*.
- Notebook tetap mandiri agar dapat dijalankan di Kaggle. Komponen dataset,
  model, dan training yang dapat digunakan kembali tersedia sebagai package
  Python di `src/lung_cnn/`.

## Persyaratan

- Python 3.10 atau lebih baru.
- Dataset citra X-Ray paru-paru dengan folder kelas `Lung_Opacity`, `Normal`,
  dan `Viral Pneumonia`.
- GPU NVIDIA disarankan untuk menjalankan seluruh eksperimen. Pelatihan di
  CPU dapat berjalan, tetapi akan jauh lebih lambat.
- Akses internet diperlukan saat model pretrained `timm` belum tersedia di
  cache lokal.

Dataset tidak disertakan di repository. Pastikan Anda mengikuti ketentuan
penggunaan dan lisensi dataset sumber.

## Instalasi lokal

Buat virtual environment, aktifkan, lalu instal proyek beserta dependensinya:

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Untuk pelatihan dengan GPU, pastikan versi PyTorch sesuai dengan driver dan
CUDA pada komputer Anda. Jika perlu, ikuti petunjuk instalasi resmi
[PyTorch](https://pytorch.org/get-started/locally/).

## Menjalankan notebook di Kaggle

1. Buka `notebooks/lung_disease_pytorch.ipynb` di Kaggle.
2. Tambahkan dataset `fatemehmehrparvar/lung-disease` melalui bagian **Input**.
   Notebook mencari tiga folder kelas di bawah `/kaggle/input`.
3. Pada **Session options**, aktifkan GPU (P100 atau T4 disarankan) dan
   **Internet** agar bobot pretrained dapat diunduh.
4. Jalankan seluruh sel notebook dari atas ke bawah.

Notebook menetapkan seed `999` untuk Python, NumPy, dan PyTorch. Hasil
checkpoint, tabel perbandingan, serta grafik disimpan secara default di
`/kaggle/working/outputs`.

### Menjalankan notebook di lingkungan lain

Notebook menggunakan `/kaggle/input` sebagai direktori input dan
`/kaggle/working/outputs` sebagai direktori output secara default. Atur variabel
lingkungan berikut sebelum menjalankan notebook untuk menggunakan lokasi lain:

```text
LUNG_CNN_INPUT_DIR=<direktori yang berisi folder dataset>
LUNG_CNN_OUTPUT_DIR=<direktori untuk hasil eksperimen>
```

Folder dataset dapat memiliki subfolder tambahan; notebook akan mencari folder
yang berisi ketiga kelas tersebut.

## Hasil yang dibuat

Selama notebook berjalan, hasil eksperimen dibuat di direktori output:

- Checkpoint bobot untuk setiap model (`.pt`).
- `pytorch_model_comparison.csv` berisi metrik perbandingan.
- `pytorch_training_curves.png` dan `pytorch_confusion_matrices.png`.

Hasil tersebut adalah keluaran proses training dan tidak disertakan ke Git.
Notebook di repository juga tidak menyimpan output eksekusi. Ekspor notebook
HTML/PDF yang tersedia secara lokal ditempatkan di `artifacts/notebook_exports/`
dan diabaikan oleh Git.

## Package dan pengujian

Package menyediakan fungsi penemuan dataset, pembuatan dataframe dan split,
`LungXRayDataset`, model `EnhancedResidualSE_CNN`, factory model `timm`, serta
helper training dan evaluasi. Notebook tetap menjadi cara utama untuk
menjalankan eksperimen lengkap; modul package dapat digunakan untuk menyusun
pipeline lokal sendiri.

Untuk menginstal alat pengembangan dan menjalankan pengujian:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Struktur repository

```text
.
├── notebooks/
│   └── lung_disease_pytorch.ipynb
├── src/
│   └── lung_cnn/
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       └── training.py
├── tests/
│   └── test_data.py
├── pyproject.toml
└── README.md
```

Data, checkpoint, direktori hasil, cache Python, dan ekspor notebook tidak
ditrack oleh Git. Lihat `.gitignore` untuk daftar pola yang diabaikan.
