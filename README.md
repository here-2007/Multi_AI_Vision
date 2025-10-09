#  Multi AI Vision Project

This repository contains four independent deep learning models combined into a single integrated system for multi-domain image classification and analysis.  

##  Repository Structure

├── models/

│ ├── object_classifier.ipynb

│ ├── plant_disease_detector.ipynb

│ ├── chest_xray_anomaly.ipynb

│ ├── face_emotion_detector.ipynb

│

├── sample_images/

│ ├── object_sample.jpg

│ ├── plant_sample.jpg

│ ├── xray_sample.jpg

│ ├── face_sample.jpg

│

├── main_interference.ipynb

├── README.md

---

##  Models Overview

### 1️ Object Classifier  
A CNN-based model that classifies common objects using pretrained architectures fine-tuned on open-source datasets.

### 2️ Plant Disease Detector  
Trained to detect and classify plant leaf diseases. Helps identify crop health from leaf images.

### 3️ Chest X-Ray Anomaly Detector  
Detects chest anomalies (e.g., viral,bacterial) from X-ray images using convolutional layers fine-tuned on **Kaggle’s Chest X-Ray dataset**.

### 4️ Face Emotion Detector  
Classifies human facial expressions into emotional categories such as *happy, sad, angry, surprised*, etc., using data from **Hugging Face emotion datasets**.

---

##  Integrated System

The file **`main_interference.ipynb`** acts as a unifying inference pipeline that loads all four trained `.h5` models and performs multi-model inference in a single run.  
It allows seamless switching between models and handles predictions for different image domains.

---

##  Training and Deployment

- Each model was trained separately using **TensorFlow/Keras**.
- Datasets were sourced from **Kaggle** and **Hugging Face**.
- The trained models (`.h5` files) were uploaded to **Kaggle** and then loaded dynamically inside `main_interference.ipynb` for inference.
- `sample_images/` directory contains test images used for validating predictions of each model.

---

##  Future Improvements
- Deploy unified API endpoint for all four models.
- Optimize inference using TensorRT / ONNX for faster processing.
- Build web-based dashboard to visualize predictions interactively.

---

##  Tech Stack
**Python, TensorFlow, Keras, NumPy, Pandas, Kaggle API, Hugging Face Datasets**

---
##  Installation
Clone the repository:

```bash
git clone <https://github.com/here-2007/Multi_AI_Vision.git>
```

## Launch Jupiter Notebook
```bash
jupyter notebook main_interference.ipynb
```
