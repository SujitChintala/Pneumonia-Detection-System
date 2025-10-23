"""
Data Preprocessing Module for Pneumonia Detection
Handles data loading, augmentation, and preparation
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import os

class DataLoader:
    """
    Data loader with augmentation for training robust models
    """
    def __init__(self):
        self.img_size = config.IMG_SIZE
        self.batch_size = config.BATCH_SIZE
        
    def create_generators(self):
        """
        Create data generators with augmentation for training
        and without augmentation for validation/testing
        """
        # Training data augmentation - critical for good generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data - only rescaling
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            config.TRAIN_DIR,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        # Load validation data
        val_generator = val_test_datagen.flow_from_directory(
            config.VAL_DIR,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Load test data
        test_generator = val_test_datagen.flow_from_directory(
            config.TEST_DIR,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def get_dataset_info(self):
        """
        Get information about the dataset
        """
        train_gen, val_gen, test_gen = self.create_generators()
        
        info = {
            'train_samples': train_gen.samples,
            'val_samples': val_gen.samples,
            'test_samples': test_gen.samples,
            'classes': train_gen.class_indices,
            'num_classes': len(train_gen.class_indices)
        }
        
        return info

if __name__ == '__main__':
    # Test the data loader
    print("Loading and preprocessing data...")
    loader = DataLoader()
    info = loader.get_dataset_info()
    
    print("\n=== Dataset Information ===")
    print(f"Training samples: {info['train_samples']}")
    print(f"Validation samples: {info['val_samples']}")
    print(f"Test samples: {info['test_samples']}")
    print(f"Classes: {info['classes']}")
    print(f"Number of classes: {info['num_classes']}")
    print("\nData preprocessing setup complete!")
