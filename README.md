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

2. Jalankan training

python train_classifier.py

3. Jalankan deteksi

python inference_classifier.py
