"""
Model Evaluation and Analysis Script
Generates comprehensive evaluation metrics and visualizations
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from data_preprocessing import DataLoader
import config
import os

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.data_loader = DataLoader()
        
    def evaluate(self):
        """
        Perform comprehensive evaluation on test set
        """
        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        # Load test data
        _, _, test_gen = self.data_loader.create_generators()
        
        # Get predictions
        print("\nGenerating predictions...")
        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes
        
        # Classification report
        print("\n" + "=" * 40)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(
            y_true, 
            y_pred, 
            target_names=config.CLASS_NAMES,
            digits=4
        ))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # ROC curve
        self.plot_roc_curve(y_true, y_pred_probs)
        
        # Sample predictions
        self.plot_sample_predictions(test_gen, y_pred_probs)
        
        print("\n✓ Evaluation complete! Check 'results' folder for visualizations.")
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot and save confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Pneumonia Detection', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
        plt.close()
        
    def plot_roc_curve(self, y_true, y_pred_probs):
        """
        Plot and save ROC curve
        """
        # Calculate ROC curve for PNEUMONIA class (class 1)
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, 
            color='darkorange', 
            lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Pneumonia Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
        plt.close()
        
    def plot_sample_predictions(self, test_gen, y_pred_probs, num_samples=9):
        """
        Plot sample predictions with confidence scores
        """
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle('Sample Predictions - Pneumonia Detection', fontsize=16, fontweight='bold')
        
        # Get a batch of images
        images, labels = next(test_gen)
        predictions = y_pred_probs[:num_samples]
        
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Display image
                ax.imshow(images[i])
                
                # Get prediction
                pred_class = np.argmax(predictions[i])
                true_class = np.argmax(labels[i])
                confidence = predictions[i][pred_class] * 100
                
                # Set title with color coding
                color = 'green' if pred_class == true_class else 'red'
                title = f"True: {config.CLASS_NAMES[true_class]}\n"
                title += f"Pred: {config.CLASS_NAMES[pred_class]} ({confidence:.1f}%)"
                ax.set_title(title, color=color, fontsize=10)
                
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(config.RESULTS_DIR, 'sample_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample predictions saved to: {save_path}")
        plt.close()

if __name__ == '__main__':
    # Evaluate the best model
    model_path = os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_best.h5')
    
    if os.path.exists(model_path):
        evaluator = ModelEvaluator(model_path)
        evaluator.evaluate()
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
