# Deteksi Bahasa Isyarat BISINDO

## Deskripsi
Sistem deteksi bahasa isyarat BISINDO menggunakan Machine Learning dan MediaPipe.

## Dataset
Dataset terdiri dari 20 kelas (0-19) dengan sekitar 200 gambar per kelas.

## Tools
- Python
- OpenCV
- MediaPipe
- Scikit-Learn

## Cara Menjalankan

1. Install library

pip install opencv-python mediapipe scikit-learn numpy

2. Mengambil Dataset

python collect_imgs.py

4. Proses Preprocessing

python Create_dataset.py

5. Jalankan training

python train_classifier.py

6. Jalankan deteksi

python inference_classifier.py

7. Menghubungkan sistem dengan web

python app.py
