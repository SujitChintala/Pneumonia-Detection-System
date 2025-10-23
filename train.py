"""
Training Script for Pneumonia Detection Model
Handles the complete training pipeline with evaluation
"""
import tensorflow as tf
from data_preprocessing import DataLoader
from model import PneumoniaDetectionModel
import config
import matplotlib.pyplot as plt
import os

class ModelTrainer:
    """
    Handles model training and evaluation
    """
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_builder = PneumoniaDetectionModel()
        self.history = None
        
    def train(self, fine_tune=True):
        """
        Train the model with optional fine-tuning
        """
        print("=" * 50)
        print("PNEUMONIA DETECTION MODEL TRAINING")
        print("=" * 50)
        
        # Load data
        print("\n[1/5] Loading and preprocessing data...")
        train_gen, val_gen, test_gen = self.data_loader.create_generators()
        info = self.data_loader.get_dataset_info()
        
        print(f"✓ Training samples: {info['train_samples']}")
        print(f"✓ Validation samples: {info['val_samples']}")
        print(f"✓ Test samples: {info['test_samples']}")
        
        # Build and compile model
        print("\n[2/5] Building model architecture...")
        self.model_builder.build_model()
        self.model_builder.compile_model()
        print("✓ Model built successfully using ResNet50")
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks()
        
        # Initial training with frozen base
        print("\n[3/5] Training model (Phase 1: Feature extraction)...")
        print(f"Epochs: {config.EPOCHS}")
        print(f"Batch size: {config.BATCH_SIZE}")
        
        history1 = self.model_builder.model.fit(
            train_gen,
            epochs=config.EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning (optional but recommended)
        if fine_tune:
            print("\n[4/5] Fine-tuning model (Phase 2: Unfreezing layers)...")
            self.model_builder.unfreeze_base_model(num_layers_to_unfreeze=50)
            
            history2 = self.model_builder.model.fit(
                train_gen,
                epochs=5,  # Fewer epochs for fine-tuning
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            self.history = {
                key: history1.history[key] + history2.history[key]
                for key in history1.history.keys()
            }
        else:
            self.history = history1.history
        
        # Final evaluation
        print("\n[5/5] Evaluating model on test set...")
        test_results = self.model_builder.model.evaluate(test_gen, verbose=1)
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED!")
        print("=" * 50)
        print("\nTest Set Results:")
        print(f"  Loss: {test_results[0]:.4f}")
        print(f"  Accuracy: {test_results[1]:.4f}")
        print(f"  Precision: {test_results[2]:.4f}")
        print(f"  Recall: {test_results[3]:.4f}")
        print(f"  AUC: {test_results[4]:.4f}")
        
        # Save final model
        final_model_path = os.path.join(
            config.MODEL_DIR, 
            f'{config.MODEL_NAME}_final.h5'
        )
        self.model_builder.model.save(final_model_path)
        print(f"\n✓ Model saved to: {final_model_path}")
        
        # Plot and save training history
        self.plot_training_history()
        
        return self.history, test_results
    
    def plot_training_history(self):
        """
        Plot and save training history
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History - Pneumonia Detection Model', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Train')
        axes[0, 1].plot(self.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history['precision'], label='Train')
        axes[1, 0].plot(self.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history['recall'], label='Train')
        axes[1, 1].plot(self.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {plot_path}")
        plt.close()

if __name__ == '__main__':
    # Train the model
    trainer = ModelTrainer()
    history, test_results = trainer.train(fine_tune=True)
    
    print("\n🎉 Training pipeline completed successfully!")
    print("Ready for deployment!")
