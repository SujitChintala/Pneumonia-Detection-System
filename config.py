"""
Configuration file for Pneumonia Detection System
"""
import os

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'pneuminia', 'chest_xray', 'chest_xray')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_CLASSES = 2

# Model architecture
MODEL_NAME = 'pneumonia_detection_model'
BACKBONE = 'resnet50'  # Using ResNet50 for transfer learning

# Training parameters
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
