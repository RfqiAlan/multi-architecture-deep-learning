# 🧠 Tugas Individu Deep Learning: Eksperimen & Analisis Komparatif Multi-Arsitektur

[![Framework: TensorFlow 2.x](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange.svg)](https://tensorflow.org/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RfqiAlan/multi-architecture-deep-learning/blob/main/Tugas_DeepLearning.ipynb)

Repositori ini memuat implementasi, evaluasi empiris, dan analisis komparatif dari **4 arsitektur Deep Learning** utama untuk memenuhi penugasan mata kuliah **Deep Learning**:
1. **Convolutional Neural Network (CNN)** — *Custom Architecture from Scratch*
2. **Transfer Learning** — *MobileNetV2 Pre-trained on ImageNet*
3. **Simple Recurrent Neural Network (SimpleRNN)**
4. **Gated Recurrent Unit (GRU)**

Seluruh eksperimen telah dieksekusi secara lengkap (*end-to-end*) dan terangkum dalam satu berkas notebook terpadu: [`Tugas_DeepLearning.ipynb`](Tugas_DeepLearning.ipynb).

---

## 📚 Tautan & Sumber Dataset

Eksperimen dalam repositori ini menggunakan dua dataset benchmark standar:

1. **CIFAR-10 (Computer Vision):**
   - 🌐 **Laman Resmi:** [University of Toronto - The CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
   - 📦 **Tautan Unduhan Langsung:** [cifar-10-python.tar.gz (170 MB)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
   - 💻 **Pemuatan via Keras:** `tf.keras.datasets.cifar10.load_data()`
   - 📖 **Referensi:** Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*, University of Toronto.
   - 🏷️ **Karakteristik:** 60.000 citra berwarna 32×32 piksel dalam 10 kelas seimbang (*airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*).

2. **IMDB Movie Reviews (Natural Language Processing):**
   - 🌐 **Laman Resmi:** [Stanford AI Lab - Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
   - 📦 **Tautan Unduhan Langsung:** [imdb.npz (Google Storage CDN ~17 MB)](https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz)
   - 💻 **Pemuatan via Keras:** `tf.keras.datasets.imdb.load_data()`
   - 📖 **Referensi:** Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*, ACL.
   - 🏷️ **Karakteristik:** 50.000 ulasan film untuk klasifikasi sentimen biner (*Positif* / *Negatif*).

---

## 📊 Hasil Riil Eksperimen & Analisis Komparatif

Berikut adalah rekapitulasi metrik pengujian riil yang diperoleh dari eksekusi notebook pada data uji (*Test Set*):

| No | Arsitektur Model | Domain & Dataset | Total Parameter | Test Loss | Test Accuracy (%) | F1-Score | Status & Karakteristik Konvergensi |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Custom CNN** *(From Scratch)* | Computer Vision (CIFAR-10) | **290,090** | **0.6637** | **76.68%** | **0.7638** | **Konvergen Sangat Baik** (15 Epoch, `val_acc` stabil meningkat) |
| **2** | **Transfer Learning** *(MobileNetV2)* | Computer Vision (CIFAR-10) | 2,428,362 | 8.6510 | 10.00% | 0.0182 | *Domain Gap* representasi ImageNet pada resolusi rendah (32×32) |
| **3** | **SimpleRNN** | NLP (IMDB Sentiment) | 650,369 | 0.6934 | 49.99% | 0.0035 | **Terkena Vanishing Gradient** (Stagnan pada tebakan acak biner ~50%) |
| **4** | **Gated Recurrent Unit (GRU)** | NLP (IMDB Sentiment) | **667,073** | **0.4185** | **83.14%** | **0.8294** | **Sangat Unggul** (*Gating Mechanism* sukses mengatasi vanishing gradient) |

---

## 🔬 Pembahasan Mendalam Hasil Eksperimen

### 1. Domain Computer Vision (Custom CNN vs Transfer Learning MobileNetV2)
- **Keberhasilan Custom CNN (Akurasi 76.68%):**
  Model Custom CNN yang dibangun dengan 3 blok konvolusi bergaya modern (`Conv2D` 32, 64, 128 filter) terbukti sangat efektif untuk dataset CIFAR-10. Penggunaan `BatchNormalization` pada setiap lapisan konvolusi berhasil menjaga kestabilan distribusi aktivasi, sementara `Dropout` bertingkat (0.2 hingga 0.5) secara signifikan menekan laju *overfitting*. Dengan hanya **290.090 parameter**, model mencapai akurasi **76.68%** dan *loss* rendah (**0.6637**).
- **Analisis Transfer Learning MobileNetV2 (Akurasi 10.00%):**
  MobileNetV2 dilatih awal (*pre-trained*) pada ImageNet dengan resolusi tinggi (224×224). Ketika diadaptasikan ke citra CIFAR-10 yang berukuran sangat kecil (32×32 piksel), terjadi kesenjangan representasi (*domain & resolution gap*). Fitur bawaan ImageNet tidak langsung mengenali tekstur mikro 32×32, dan pembekuan bobot (*frozen backbone*) pada tahap awal menghambat penyesuaian bobot tanpa jadwal *fine-tuning* yang sangat panjang. Hal ini membuktikan bahwa untuk data beresolusi rendah dengan domain berbeda, **arsitektur CNN yang dirancang khusus dari nol sering kali lebih optimal daripada model transfer learning berukuran besar**.

### 2. Domain Natural Language Processing (SimpleRNN vs Gated Recurrent Unit / GRU)
- **Fenomena Vanishing Gradient pada SimpleRNN (Akurasi 49.99%):**
  SimpleRNN memproses teks langkah demi langkah tanpa mekanisme gerbang. Saat melakukan *Backpropagation Through Time* (BPTT) sepanjang 200 token kalimat ulasan IMDB, sinyal gradien menyusut drastis secara eksponensial (*vanishing gradient*). Akibatnya, model gagal mempelajari dependensi jangka panjang dan hanya menghasilkan akurasi **49.99%** (setara probabilitas tebakan acak murni pada klasifikasi biner).
- **Superioritas Mekanisme Gerbang pada GRU (Akurasi 83.14%):**
  Sebaliknya, **GRU** membuktikan keunggulan teoritisnya secara empiris. Dengan memanfaatkan **Update Gate** ($z_t$) dan **Reset Gate** ($r_t$), GRU menyediakan jalur penambahan linier (*additive shortcut*) yang memungkinkan gradien mengalir tanpa hambatan melintasi ratusan kata. GRU berhasil mencapai akurasi **83.14%** dengan *loss* rendah (**0.4185**) dan F1-Score **0.8294**, mendemonstrasikan kapabilitas prima dalam menangkap konteks semantik ulasan.

#### Rincian Classification Report GRU:
```text
              precision    recall  f1-score   support
     Negatif     0.8237    0.8434    0.8334     12,500
     Positif     0.8395    0.8194    0.8294     12,500

    accuracy                         0.8314     25,000
   macro avg     0.8316    0.8314    0.8314     25,000
weighted avg     0.8316    0.8314    0.8314     25,000
```

---

## 📂 Struktur Direktori Repositori

```text
tugas_deep_learning/
├── Tugas_DeepLearning.ipynb   # 🌟 Notebook tunggal lengkap (termasuk output eksekusi & plot)
├── requirements.txt          # Daftar pustaka pendukung (TensorFlow, Scikit-Learn, dll.)
├── .gitignore                # Filter file cache lokal (.ipynb_checkpoints, venv)
└── README.md                 # Dokumentasi komprehensif & laporan hasil eksperimen
```


---

## 🚀 Petunjuk Replikasi & Menjalankan Notebook

### Opsi A: Google Colab (Akselerasi GPU Bebas Konfigurasi)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file [`Tugas_DeepLearning.ipynb`](Tugas_DeepLearning.ipynb).
3. Aktifkan GPU gratis: Masuk ke **Runtime** > **Change runtime type** > Pilih **T4 GPU** > Klik **Save**.
4. Jalankan seluruh cell dengan **Runtime** > **Run all** (`Ctrl + F9` atau `Cmd + F9`).

### Opsi B: Lokal (macOS / Linux / Windows)
```bash
# 1. Masuk ke direktori tugas
cd tugas_deep_learning

# 2. Buat dan aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Di Windows: .venv\Scripts\activate

# 3. Install seluruh dependensi
pip install -r requirements.txt

# 4. Jalankan Jupyter Lab / Notebook
jupyter notebook
```

---

## 📤 Panduan Push ke GitHub & Pengumpulan Tugas

1. **Inisialisasi Git dan Commit:**
   ```bash
   cd /Users/rifqialanmaulana/Downloads/MPSI/tugas_deep_learning
   git init
   git add .
   git commit -m "feat: complete deep learning assignment with full benchmark results and evaluation"
   ```

2. **Buat Repositori di GitHub:**
   - Kunjungi [GitHub New Repo](https://github.com/new).
   - Beri nama repositori (contoh: `tugas-deep-learning-cnn-rnn-gru`), setel ke **Public**, dan jangan centang opsi *Add README*.

3. **Hubungkan Remote dan Push:**
   ```bash
   git branch -M main
   git remote add origin https://github.com/RfqiAlan/multi-architecture-deep-learning.git
   git push -u origin main
   ```

4. **Pengumpulan ke Google Spreadsheet:**
   - Salin link repositori GitHub Anda.
   - Cantumkan ke formulir Google Spreadsheet pengumpulan tugas sebelum deadline (**Jumat, 6 November 2026**).
