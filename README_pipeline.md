# Panduan Menjalankan Pipeline Kilometer (YOLO + OCR)

Dokumen ini menjelaskan cara menggunakan dan menjalankan script `pipeline.py` untuk mendeteksi serta mengenali angka **stand meter** dan **nomor meter** secara langsung dari satu foto meteran utuh.

---

## 📋 Persyaratan System & Library
Pastikan library berikut sudah terinstal di environment Python Anda:
```bash
pip install ultralytics opencv-python torch torchvision pillow lmdb natsort
```

---

## 🚀 Cara Menjalankan via Command Line (CLI)

Untuk menjalankan deteksi secara langsung dari terminal, jalankan perintah berikut:

```bash
python pipeline.py \
  --image_path <path_ke_foto_kilometer> \
  --yolo_model <path_ke_model_yolo.pt> \
  --saved_model <path_ke_model_ocr.pth>
```

### 💡 Contoh Penggunaan Nyata:
```bash
python pipeline.py \
  --image_path demo_image/demo_10.jpg \
  --yolo_model best.pt \
  --saved_model saved_models/TPS-Custom-BiLSTM-CTC-Seed1111/best_accuracy.pth
```

### ⚙️ Parameter Penting yang Dapat Ditambahkan:

| Parameter | Default | Deskripsi |
| :--- | :--- | :--- |
| `--image_path` | *(Wajib)* | Jalur menuju foto kilometer asli yang akan diproses. |
| `--yolo_model` | `best.pt` | Jalur ke weights model YOLO hasil training Anda. |
| `--saved_model` | *(Wajib)* | Jalur ke weights model OCR (`best_accuracy.pth`). |
| `--save_crops_dir` | `None` | Jika diisi dengan folder (misal: `--save_crops_dir output_crop`), hasil crop teks YOLO akan disimpan. |
| `--padding` | `8` | Ukuran margin ekstra (pixel) di sekeliling teks untuk membantu akurasi OCR. |
| `--yolo_conf` | `0.25` | Ambang batas minimum confidence score untuk deteksi box YOLO. |

---

## 🐍 Cara Menggunakan di Script Python Lain

Anda juga bisa mengintegrasikan detektor ini ke dalam program Python Anda yang lain:

```python
from pipeline import KilometerPipeline
import argparse

# 1. Definisikan opsi OCR sesuai arsitektur model Anda
# Parameter default di bawah sudah disesuaikan dengan arsitektur latihan Anda (TPS-CustomAttentionCNN-BiLSTM-CTC)
opt = argparse.Namespace(
    batch_max_length=25,
    imgH=64,
    imgW=200,
    rgb=False,
    character="0123456789",
    sensitive=False,
    PAD=False,
    Transformation="TPS",
    FeatureExtraction="CustomAttentionCNN",
    SequenceModeling="BiLSTM",
    Prediction="CTC"
)

# 2. Inisialisasi Pipeline
pipeline = KilometerPipeline(
    yolo_model_path="best.pt",
    ocr_model_path="saved_models/TPS-Custom-BiLSTM-CTC-Seed1111/best_accuracy.pth",
    ocr_opt=opt
)

# 3. Jalankan pemrosesan gambar
hasil = pipeline.process(
    image_path="demo_image/demo_10.jpg",
    padding=8,
    conf_threshold=0.25,
    save_crops_dir="hasil_pemotongan"  # Opsional: Simpan crop gambar untuk verifikasi
)

# 4. Ambil output
print("Stand Meter  :", hasil['meter']['text'], f"(Conf: {hasil['meter']['conf']:.4f})")
print("Nomor Meter  :", hasil['nomor_meter']['text'], f"(Conf: {hasil['nomor_meter']['conf']:.4f})")
```

---

## 🔍 Cara Kerja Pemisahan Stand & Nomor Meter
* **Urutan Vertikal**: YOLO mendeteksi box teks. Box tersebut diurutkan dari atas ke bawah berdasarkan koordinat Y.
  * Box paling **atas** secara otomatis ditetapkan sebagai **Stand Meter** (`meter`).
  * Box paling **bawah** secara otomatis ditetapkan sebagai **Nomor Meter** (`nomor_meter`).
* **Kondisi 1 Box**: Jika YOLO hanya mendeteksi 1 box, script akan membandingkan titik tengah box dengan titik tengah gambar asli. Jika berada di atas garis horizontal tengah, dianggap `meter`. Jika di bawah, dianggap `nomor_meter`.
