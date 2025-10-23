# 🚀 Quick Start Guide

## Get Started in 3 Steps!

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train.py
```
⏱️ Training takes approximately 30-60 minutes (depending on hardware)

### Step 3: Run the Application
```bash
python app.py
```
🌐 Open http://localhost:5000 in your browser

---

## 🧪 Test Single Image

Quickly test a single X-ray image:
```bash
python predict.py path/to/xray.jpg
```

---

## 📊 Evaluate Model Performance

Generate comprehensive evaluation metrics:
```bash
python evaluate.py
```

This creates:
- Confusion matrix
- ROC curve
- Sample predictions
- Classification report

---

## 🛠️ Troubleshooting

### Issue: Model not found
**Solution**: Train the model first using `python train.py`

### Issue: TensorFlow installation error
**Solution**: 
```bash
pip install tensorflow==2.13.0 --upgrade
```

### Issue: Memory error during training
**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # instead of 32
```

### Issue: Flask server won't start
**Solution**: Make sure port 5000 is available:
```bash
# Windows
netstat -ano | findstr :5000

# Linux/Mac
lsof -i :5000
```

---

## 📈 Expected Performance

| Phase | Time | GPU | CPU |
|-------|------|-----|-----|
| Training | ~30 min | 8GB VRAM | ~2 hours |
| Inference | <1 sec | Any | Any |
| Evaluation | ~2 min | Any | Any |

---

## 💡 Tips for Best Results

1. **Training**:
   - Use GPU if available (10x faster)
   - Monitor training with TensorBoard: `tensorboard --logdir results/logs`
   - Let the model complete all epochs for best results

2. **Deployment**:
   - Use the `_best.h5` model for production
   - Implement proper error handling
   - Add authentication for production use

3. **Testing**:
   - Test with various X-ray qualities
   - Verify predictions with medical professionals
   - Use ensemble methods for critical decisions

---

## 📞 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review code comments for implementation details
- Test each component separately to isolate issues
