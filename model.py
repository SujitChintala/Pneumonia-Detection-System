"""
Model Architecture for Pneumonia Detection
Uses Transfer Learning with ResNet50 for robust performance
"""
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import config
import os
from datetime import datetime

class PneumoniaDetectionModel:
    """
    CNN Model using Transfer Learning for Pneumonia Detection
    Architecture: ResNet50 + Custom Classification Head
    """
    def __init__(self):
        self.img_size = config.IMG_SIZE + (3,)  # (224, 224, 3)
        self.num_classes = config.NUM_CLASSES
        self.model = None
        
    def build_model(self):
        """
        Build the model using transfer learning with ResNet50
        """
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_size
        )
        
        # Freeze the base model layers initially
        base_model.trainable = False
        
        # Build custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=outputs)
        
        return self.model
    
    def compile_model(self):
        """
        Compile the model with optimizer, loss, and metrics
        """
        self.model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
    def get_callbacks(self):
        """
        Define callbacks for training
        """
        # Model checkpoint - save best model
        checkpoint_path = os.path.join(
            config.MODEL_DIR, 
            f'{config.MODEL_NAME}_best.h5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
        
        # TensorBoard
        log_dir = os.path.join(
            config.RESULTS_DIR, 
            'logs', 
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        return [checkpoint, early_stop, reduce_lr, tensorboard]
    
    def unfreeze_base_model(self, num_layers_to_unfreeze=50):
        """
        Unfreeze the last n layers of base model for fine-tuning
        """
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the last n
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
    
    def summary(self):
        """
        Print model summary
        """
        return self.model.summary()

if __name__ == '__main__':
    print("Building Pneumonia Detection Model...")
    model_builder = PneumoniaDetectionModel()
    model = model_builder.build_model()
    model_builder.compile_model()
    
    print("\n==== Model Architecture ====")
    model_builder.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print("Model built successfully!")
