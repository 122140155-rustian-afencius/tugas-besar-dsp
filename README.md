# Final Project Mata Kuliah Digital Processing Signal (IF3024)

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T., Ph.D**

# **Real Time rPPG & Respiration Signal**

## **Anggota Kelompok**

| **Nama**                    | **NIM**   | **ID GITHUB**                                                               |
| --------------------------- | --------- | --------------------------------------------------------------------------- |
| Rustian Afencius Marbun | 122140155 | <a href="https://github.com/122140155-rustian-afencius">@122140155-rustian-afencius</a> |
| Shintya Ayu Wardani     | 122140138 | <a href="https://github.com/shintyawa">@shintyawa</a>                     |

---

## **Deskripsi Proyek**

Proyek ini mengembangkan **sistem monitoring vital sign non-invasive** yang revolusioner dengan memanfaatkan teknologi computer vision dan digital signal processing untuk mendeteksi detak jantung dan laju pernapasan secara real-time hanya menggunakan webcam standar. Sistem ini menggabungkan teknik **Remote Photoplethysmography (rPPG)** dan **pose estimation** untuk memberikan solusi monitoring kesehatan yang mudah diakses tanpa memerlukan sensor fisik yang menyentuh tubuh.

### **Teknologi yang Digunakan**

**Deteksi Detak Jantung (rPPG):**
- Menggunakan metode **Plane Orthogonal-to-Skin (POS)** untuk mengekstrak sinyal rPPG dari perubahan intensitas warna pada area dahi
- Menerapkan **Butterworth band-pass filter** (0.7-3.5 Hz) untuk mengisolasi frekuensi detak jantung
- Implementasi **wavelet denoising** dan **Savitzky-Golay smoothing** untuk meningkatkan kualitas sinyal
- Analisis frekuensi menggunakan **Fast Fourier Transform (FFT)** untuk menghitung beats per minute (BPM)

**Deteksi Laju Pernapasan:**
- Memanfaatkan **MediaPipe Pose** untuk mendeteksi landmark tubuh dengan akurasi tinggi
- Menganalisis pergerakan vertikal bahu yang mencerminkan ekspansi dan kontraksi dada saat bernapas
- Penerapan **Butterworth band-pass filter** (0.1-0.5 Hz) untuk mengisolasi frekuensi pernapasan normal
- Kalkulasi respiration rate melalui analisis spektral domain frekuensi

### **Fitur Unggulan**

- **Real-time Processing**: Monitoring simultan detak jantung dan pernapasan dengan update grafik langsung
- **Modular Architecture**: Desain kode yang terstruktur dengan komponen terpisah untuk pose detection, signal processing, analysis, dan visualization
- **Adaptive Quality Control**: Sistem penilaian kualitas sinyal otomatis dengan confidence scoring
- **Interactive GUI**: Antarmuka pengguna yang intuitif dengan kontrol filter yang dapat disesuaikan
- **Signal Visualization**: Tampilan multi-panel untuk raw signal, filtered signal, dan spektral analysis
- **Non-invasive Monitoring**: Tidak memerlukan kontak fisik atau perangkat khusus selain webcam

### **Tujuan & Manfaat**

Proyek ini bertujuan untuk **mendemokratisasi akses monitoring kesehatan** dengan menyediakan teknologi yang:
- Dapat digunakan di rumah tanpa peralatan medis mahal
- Mendukung telemedicine dan konsultasi jarak jauh
- Memberikan monitoring kontinyu untuk pasien dengan kondisi kronis
- Menjadi proof-of-concept untuk aplikasi healthcare berbasis AI

Sistem ini membuktikan potensi besar **teknologi computer vision dalam bidang kesehatan digital**, membuka jalan untuk pengembangan aplikasi monitoring vital sign yang lebih canggih dan accessible untuk masyarakat luas.

---

## **Alat yang Digunakan**
|**Logo**  | **Nama**   | **Fungsi**   |
| -------- |------------|--------------|
|<img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="VS Code Logo" width="70">|   VSCode   |Editor kode utama untuk menulis, mengedit, dan menjalankan skrip Python dengan fitur seperti debugging, ekstensi, dan integrasi Git.|
|<img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="70">|   Python   |Sebagai bahasa pemrograman utama untuk membangun dan menjalankan logika pemrosesan sinyal, analisis data, serta antarmuka sistem.|
---
## **Library yang Digunakan**

Berikut adalah daftar library Python yang digunakan dalam proyek ini, beserta fungsinya:

- - -

| **No.** | **Library**                             | **Fungsi**                                                                 |
|-----|-------------------------------------|------------------------------------------------------------------------|
| 1   | `logging`                           | Mencatat log informasi, peringatan, dan kesalahan saat eksekusi.      |
| 2   | `sys`                               | Mengakses fungsi sistem; keluar saat terjadi kesalahan kritis.        |
| 3   | `tkinter`                           | Library GUI untuk membuat jendela, tombol, dan grafik.                |
| 4   | `cv2` (OpenCV)                      | Mengambil dan memproses video secara real-time dari kamera.           |
| 5   | `numpy`                             | Komputasi numerik dan manipulasi array.                               |
| 6   | `matplotlib.pyplot`                 | Visualisasi grafik sinyal secara real-time.                           |
| 7   | `threading`                         | Menjalankan proses paralel tanpa mengganggu GUI.                      |
| 8   | `queue`                             | Menyediakan antrean data antar-thread secara aman.                    |
| 9   | `time`                              | Mengatur delay dan mencatat waktu.                                    |
| 10  | `PIL` (Pillow)                      | Mengubah frame ke format gambar untuk ditampilkan di `tkinter`.      |
| 11  | `collections.deque`                | Buffer efisien untuk menyimpan data sinyal.                           |
| 12  | `mediapipe`                         | Deteksi pose dan wajah untuk ekstraksi landmark.                      |
| 13  | `scipy`                             | Pemrosesan sinyal seperti filtering.                                  |
| 14  | `pywt` (PyWavelets)                 | Denoising sinyal dengan transformasi wavelet.                         |
| 15  | `dataclasses`                       | Membuat class data dengan sintaks ringkas.                            |
| 16  | `typing`                            | Memberikan type hints untuk keterbacaan kode.                         |
| 17  | `asttokens`, `executing`, `stack-data`, `pure-eval` | Mendukung pelacakan eksekusi dan debugging stack trace.  |
| 18  | `colorama`, `prompt-toolkit`, `pygments`, `wcwidth` | Pewarnaan teks dan tampilan interaktif di terminal. |
| 19  | `comm`, `ipykernel`, `ipython`, `jupyter-client`, `jupyter-core`, `nest-asyncio`, `matplotlib-inline` | Digunakan untuk lingkungan Jupyter Notebook. |
| 20  | `debugpy`                           | Debugger Python yang dipakai oleh VS Code.                            |
| 21  | `decorator`                         | Mempermudah pembuatan dekorator fungsi.                               |
| 22  | `exceptiongroup`                    | Menangani banyak exception secara bersamaan.                          |
| 23  | `jedi`, `parso`                     | Auto-completion dan parsing kode, digunakan di editor.                |
| 24  | `packaging`, `platformdirs`         | Mengelola metadata paket dan direktori konfigurasi.                   |
| 25  | `psutil`                            | Memantau penggunaan resource sistem.                                  |
| 26  | `pywin32`                           | Binding ke API Windows.                                               |
| 27  | `pyzmq`, `tornado`, `traitlets`     | Komunikasi jaringan untuk backend Jupyter.                            |
| 28  | `six`                               | Kompatibilitas lintas versi Python.                                   |
| 29  | `typing-extensions`                 | Mendukung anotasi tipe baru pada Python versi lama.                   |

---
## **Logbook Mingguan**

### **Minggu 1 (12 Mei 2025 - 18 Mei 2025)**
- Inisialisasi Project, membuat repositori untuk manajemen final
- Melakukan pembagian tugas antara Rustian Afencius Marbun(122140153) dan Shintya Ayu Wardani(122140138)
---
### **Minggu 2 (19 Mei 2025 - 25 Mei 2025)**
- Membuat file kode rPPG dan Respiration Signal
---
### **Minggu 3 (26 Mei 2025 - 31 Mei 2025)**
- Membuat laporan menggunakan overleaf
- Melakukan Fiksasi Kode rPPG dan Respiration Signal
- Menggabungkan Kode rPPG dan Respiration Signal ke main.py
- Melakukan refactoring code
- Merapikan folder DSP
---
## Cara Menjalankan Program

Dengan asumsi Anda sudah menginstal Python (disarankan versi 3.10), ikuti langkah-langkah berikut untuk menjalankan proyek ini:

### 1. Clone Repository

```yaml
git clone https://github.com/username/nama-repo.git
```

### 2.Membuat Virtual Environment (venv)

```yaml
python -m venv venv
```

Aktifkan environman (pilih sesuai sistem operasi Anda):
- Windows :

```yaml
venv\Scripts\activate
```
- Linux :

```yaml
source venv/bin/activate
```
### 3. Install requirements.txt

```yaml
pip install -r requirements.txt
```

### 4. Jalankan program.

```yaml
python main.py
```