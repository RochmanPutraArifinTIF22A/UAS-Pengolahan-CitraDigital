# UAS-Pengolahan-CitraDi
# Step 1: Install necessary libraries
!pip install ultralytics  # Install YOLOv8
!pip install matplotlib opencv-python-headless
!pip install roboflow

# Step 2: Import libraries
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files
import cv2
import numpy as np

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="p9AE4cfyWZVtKr7MenmF")
# project = rf.workspace("testingws").project("object-detection-bbaki")
# version = project.version(3)
# dataset = version.download("yolov8")

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="p9AE4cfyWZVtKr7MenmF")
project = rf.workspace("citradigital").project("hewandetection")
version = project.version(1)
dataset = version.download("yolov8")

import os

# Lihat folder tempat dataset diunduh
dataset_location = dataset.location  # dari RoboFlow download
print("Dataset downloaded to:", dataset_location)

from ultralytics import YOLO

# Buat model YOLOv8 baru
model = YOLO("yolov8n.pt")  # "yolov8n.pt" adalah versi YOLOv8 Nano

# Jalankan pelatihan dengan dataset
model.train(data="/content/HewanDetection-1/data.yaml", epochs=80, imgsz=640)

from ultralytics import YOLO

# Muat model terlatih
model = YOLO("/content/runs/detect/train/weights/best.pt")

dataset = version.download("yolov8")

result = model.predict(source="/content/HewanDetection-1/valid/images", imgsz=640, save=True)

# Menampilkan semua hasil prediksi
import matplotlib.pyplot as plt

# Loop melalui semua gambar hasil prediksi
for image_result in result:
    #plot setiap hasil prediksi
    image_path_with_predictions = image_result.plot() # mengembalikan array gambar

    # Tampilkan menggunakan Mathlotlib
    plt.imshow(image_path_with_predictions)
    plt.axis("off")
    plt.show()

# Step 3: Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Step 4: Load YOLO model
model = YOLO('yolov8n.pt')  # Use a pre-trained YOLOv8 model (nano version for speed)

# Step 5: Perform object detection
results = model(image_path)  # Run inference on the uploaded image

# Step 6: Visualize results
# Save the annotated image
annotated_img = results[0].plot()  # Create an annotated image (numpy array)

cv2.imwrite("/content/runs/detect/train/F1_curve.png", annotated_img)

import cv2
import matplotlib.pyplot as plt

# Loop melalui semua hasil prediksi
for image_result in result:
    # Ambil gambar yang sudah dianotasi
    annotated_img = image_result.plot()  # Mengembalikan array gambar dengan bounding box

    # Tampilkan gambar menggunakan Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLO Detected Objects")
    plt.show()
