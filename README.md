# Facial Expression Recognition System

## 📌 Project Overview
The **Facial Expression Recognition System** is a deep learning-based project designed to classify human facial expressions into different categories. This system is built using **deep convolutional neural networks (CNNs)** and incorporates **real-time facial recognition** techniques. The project is applicable in **human-computer interaction, emotional analysis, and intelligent surveillance**.

---

## 🚀 Key Features
✅ **Face Detection** – Implements Haar cascade classifiers for face detection using OpenCV.  
✅ **Deep Learning Model** – Uses **Mini-Xception** for high-accuracy facial expression recognition.  
✅ **Real-Time Processing** – Capable of detecting and classifying emotions in real-time video streams.  
✅ **Multi-Class Emotion Classification** – Recognizes **seven fundamental emotions**: Angry, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.  
✅ **Image Preprocessing** – Performs grayscale conversion, normalization, and data augmentation for improved accuracy.  
✅ **User-Friendly Interface** – Built with PyQt5 for an interactive GUI.

---

## 🛠 Technology Stack
### **Core Technologies**
- **Python** – Main programming language.
- **OpenCV** – Used for face detection and image preprocessing.
- **Keras & TensorFlow** – Deep learning frameworks for training and inference.
- **PyQt5** – GUI development for interactive user interface.

### **Model Architecture**
- **Mini-Xception** – Lightweight deep convolutional neural network for facial expression classification.
- **Batch Normalization & ReLU** – Enhances training stability and model performance.
- **Softmax Activation** – Generates probability distributions for multi-class classification.

---

## 📥 Installation Guide
### 1️⃣ Environment Preparation
Ensure you have the following dependencies installed:
```bash
pip install tensorflow keras opencv-python numpy matplotlib PyQt5
```

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo.git
cd your-repo
```

### 3️⃣ Download & Prepare Dataset
- The system uses the **FER2013 dataset**.
- Preprocess the dataset using **grayscale conversion, normalization, and data augmentation**.

### 4️⃣ Train the Model
Run the following command to start training:
```bash
python train_model.py
```

---

## 🎯 Running Guide
### ✅ Start Face Detection & Expression Recognition
Run the application with:
```bash
python run_recognition.py
```
### ✅ Real-Time Recognition
- Open the application GUI.
- Click **Start Camera** to begin real-time facial expression recognition.

---

## 🔹 Project Functionalities
- **📡 Real-Time Detection** – Processes video frames to detect faces and classify emotions.
- **📊 Model Training & Evaluation** – Provides loss and accuracy metrics during training.
- **📸 Image-Based Recognition** – Allows users to upload images for expression analysis.
- **🖥 Interactive GUI** – Provides an intuitive user experience for testing and visualization.

---

💡 **Explore the source code and implement AI-driven facial expression recognition!** 🚀

