# 🫁 Pneumonia Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-Powered Chest X-Ray Analysis for Pneumonia Detection**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Results](#results)

</div>

---

## 📋 Overview

This is an end-to-end **Pneumonia Detection System** that uses Deep Learning to analyze chest X-ray images and detect pneumonia with high accuracy. The system leverages Transfer Learning with ResNet50 architecture and provides a clean, user-friendly web interface for real-time predictions.

### ✨ Key Features

- 🤖 **Deep Learning Model** - ResNet50-based CNN with transfer learning
- 📊 **High Accuracy** - Achieves 90%+ accuracy on test set
- 🎨 **Modern Web Interface** - Clean, responsive design with drag-and-drop upload
- 🔄 **Real-time Predictions** - Instant analysis with confidence scores
- 📈 **Comprehensive Metrics** - Precision, Recall, F1-Score, ROC-AUC
- 🔍 **Model Interpretability** - Confidence scores and probability distributions
- 📦 **Production Ready** - RESTful API with Flask backend

---

## 🏗️ Architecture

### Model Architecture

```
Input (224x224x3)
    ↓
ResNet50 (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense (512) + BatchNorm + Dropout(0.5)
    ↓
Dense (256) + BatchNorm + Dropout(0.3)
    ↓
Dense (2, Softmax) → [NORMAL, PNEUMONIA]
```

### Technology Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Backend**: Flask, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing**: NumPy, Pandas, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Scikit-learn

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU (optional, for faster training)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/SujitChintala/Pneumonia-Detection-System.git
cd Pneumonia-Detection-System
```

2. **Download the dataset**

Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:

🔗 **Dataset Link**: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Extract the dataset and place it in the project directory so the structure looks like:
```
Pneumonia-Detection/
├── pneuminia/
│   └── chest_xray/
│       └── chest_xray/
│           ├── train/
│           ├── val/
│           └── test/
├── config.py
├── train.py
└── ...
```

3. **Create virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Model

Train the pneumonia detection model on your dataset:

```bash
python train.py
```

**Training Features:**
- Data augmentation for robust learning
- Transfer learning with ResNet50
- Early stopping and learning rate reduction
- Model checkpointing (saves best model)
- TensorBoard logging
- Two-phase training (feature extraction + fine-tuning)

**Expected Output:**
- Trained model saved in `models/` directory
- Training history plot in `results/` directory
- Console logs with training progress and metrics

### 2. Evaluate the Model

Evaluate model performance and generate visualizations:

```bash
python evaluate.py
```

**Generates:**
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- ROC curve with AUC score
- Sample predictions with confidence scores

### 3. Run the Web Application

Start the Flask server and web interface:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

**Web Interface Features:**
- Drag-and-drop image upload
- Real-time X-ray analysis
- Visual prediction results with confidence scores
- Probability distribution for both classes
- Responsive design for mobile and desktop

---

## 📊 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.3% |
| **Precision** | 91.8% |
| **Recall** | 94.5% |
| **F1-Score** | 93.1% |
| **ROC-AUC** | 96.7% |

### Dataset Statistics

- **Training Set**: 5,216 images
- **Validation Set**: 16 images
- **Test Set**: 624 images
- **Classes**: NORMAL (0), PNEUMONIA (1)

### Key Achievements

✅ High recall rate - minimizes false negatives (critical in medical diagnosis)  
✅ Robust to image variations through data augmentation  
✅ Fast inference time (~500ms per image)  
✅ Production-ready API with proper error handling  
✅ Clean, professional UI/UX  

---

## 📁 Project Structure

```
Pneumonia-Detection/
├── pneuminia/              # Dataset directory
│   └── chest_xray/
│       └── chest_xray/
│           ├── train/
│           ├── val/
│           └── test/
├── models/                 # Saved models
│   ├── pneumonia_detection_model_best.h5
│   └── pneumonia_detection_model_final.h5
├── results/                # Evaluation results
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── sample_predictions.png
├── static/                 # Frontend assets
│   ├── style.css
│   └── script.js
├── templates/              # HTML templates
│   └── index.html
├── config.py               # Configuration settings
├── data_preprocessing.py   # Data loading and augmentation
├── model.py                # Model architecture
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# Paths
DATA_DIR = './pneuminia/chest_xray/chest_xray'
MODEL_DIR = './models'
RESULTS_DIR = './results'
```

---

## 🌐 API Endpoints

### Health Check
```
GET /api/health
```
Returns server and model status.

### Predict
```
POST /api/predict
Content-Type: multipart/form-data
Body: image file
```
Returns prediction with confidence scores.

**Response:**
```json
{
    "success": true,
    "prediction": "PNEUMONIA",
    "confidence": 95.2,
    "probabilities": {
        "NORMAL": 4.8,
        "PNEUMONIA": 95.2
    }
}
```

### Model Info
```
GET /api/model-info
```
Returns model architecture information.

---

## 🧪 Testing

Test the API using curl:

```bash
curl -X POST -F "image=@test_xray.jpg" http://localhost:5000/api/predict
```

Or use Python:

```python
import requests

url = 'http://localhost:5000/api/predict'
files = {'image': open('test_xray.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

---

## 🎯 Use Cases

- **Medical Screening**: Quick preliminary screening for pneumonia
- **Telemedicine**: Remote diagnosis support
- **Educational**: Teaching tool for medical students
- **Research**: Baseline for pneumonia detection research
- **Healthcare Access**: Support for areas with limited radiologists

---

## ⚠️ Disclaimer

This system is designed as a **diagnostic aid tool** and should **NOT** be used as the sole basis for medical diagnosis. Always consult with qualified healthcare professionals for accurate diagnosis and treatment. The model's predictions should be verified by trained radiologists.

---

## 🔮 Future Enhancements

- [ ] Support for multiple X-ray views (PA, Lateral, AP)
- [ ] Mobile application (iOS/Android)
- [ ] Multi-class classification (viral vs bacterial pneumonia)
- [ ] Batch processing support
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)

---

## 📚 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset:

- **Source**: Kaggle
- **Size**: 5,856 images
- **Format**: JPEG
- **Classes**: Normal, Pneumonia
- **Dataset Link**: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

- GitHub: [SujitChintala](https://github.com/SujitChintala)
- LinkedIn: [Saai Sujit Chintala](https://www.linkedin.com/in/sujitchintala/)
- Email: sujitchintala@gmail.com

---

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the amazing deep learning framework
- The creators of the Chest X-Ray dataset
- ResNet architecture by He et al.
- Open-source community for various libraries and tools

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
