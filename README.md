# 🌍 Romance Language Classifier

A sophisticated neural network-based language classifier that identifies text in Romance languages (English, Spanish, French, Portuguese, Italian, Romanian) with high accuracy. Built with PyTorch and deployed as an interactive web application.

## 🚀 Live Demo

**[Try the Live App](https://language-classifier.streamlit.app/)**

## ✨ Features

- **High Accuracy**: 96.2% test accuracy across 7 language categories
- **Advanced Architecture**: Attention-based neural network with residual connections
- **Real-time Classification**: Instant language prediction with confidence scores
- **Beautiful UI**: Modern, responsive interface with interactive visualizations
- **Comprehensive Metrics**: Detailed performance analysis and confusion matrices
- **Multi-language Support**: Handles 6 Romance languages + English + Unknown category

## 🏗️ Architecture

### Model Design
- **Input**: N-gram features extracted from text
- **Architecture**: Multi-head attention layers with residual connections
- **Training**: AdamW optimizer with learning rate scheduling
- **Regularization**: Dropout, gradient clipping, and early stopping

### Technical Stack
- **Backend**: PyTorch, NumPy, Scikit-learn
- **Frontend**: Streamlit, Plotly
- **Deployment**: Streamlit Cloud, Hugging Face Hub
- **Monitoring**: TensorBoard logging

## 📊 Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.2% |
| Vocabulary Size | ~50,000 n-grams |
| Training Time | ~30 minutes |
| Inference Speed | <100ms |

### Per-Language Performance
- **English**: 98.1% F1-score
- **Spanish**: 95.8% F1-score  
- **French**: 96.3% F1-score
- **Portuguese**: 94.7% F1-score
- **Italian**: 95.2% F1-score
- **Romanian**: 93.9% F1-score

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/evan_thoms/romance-classifier.git
cd romance-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/streamlit.py
```

### Training Your Own Model
```bash
# Train the model
python src/train.py

# Monitor training with TensorBoard
tensorboard --logdir runs/
```

## 📁 Project Structure

```
romance_classifier/
├── src/
│   ├── model.py              # Neural network architectures
│   ├── train.py              # Training pipeline
│   ├── streamlit.py          # Web application
│   ├── data_loader.py        # Data processing
│   ├── feature_extraction.py # N-gram feature extraction
│   ├── eval.py              # Evaluation utilities
│   └── predict.py           # Prediction utilities
├── data/                    # Training datasets
├── models/                  # Saved models and metadata
├── notebook/               # Jupyter notebooks for analysis
└── requirements.txt        # Python dependencies
```

## 🧠 How It Works

### 1. Feature Extraction
- Converts text into n-gram features (4-grams)
- Creates vocabulary from training data
- Normalizes feature vectors

### 2. Neural Network Processing
- Projects features to high-dimensional space
- Applies multi-head attention mechanisms
- Uses residual connections for better gradient flow
- Final classification layer outputs language probabilities

### 3. Training Process
- **Data Sources**: Wikipedia articles, TED Talk translations, common phrases
- **Optimization**: AdamW with weight decay
- **Regularization**: Dropout, gradient clipping, early stopping
- **Monitoring**: TensorBoard logging for metrics

## 📈 Model Improvements

### Version 2.0 Enhancements
- ✅ Attention-based architecture
- ✅ Residual connections
- ✅ Advanced regularization techniques
- ✅ Comprehensive evaluation metrics
- ✅ Professional UI/UX design
- ✅ Real-time confidence visualization

### Planned Features
- 🔄 Transformer-based architecture
- 🔄 Multi-language sentence embeddings
- 🔄 API endpoint for integration
- 🔄 Mobile app version
- 🔄 Real-time learning capabilities

## 🎯 Use Cases

- **Content Moderation**: Automatically categorize multilingual content
- **Language Learning**: Identify language of unknown text
- **Translation Services**: Pre-processing for translation pipelines
- **Research**: Linguistic analysis and language detection studies
- **Education**: Language learning applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: Wikipedia, TED Talks parallel corpus
- **Libraries**: PyTorch, Streamlit, Hugging Face
- **Inspiration**: Natural language processing research community

## 📞 Contact

- **GitHub**: [@evan_thoms](https://github.com/evan_thoms)
- **Project Link**: [https://github.com/evan_thoms/romance-classifier](https://github.com/evan_thoms/romance-classifier)

---

⭐ **Star this repository if you found it helpful!**
