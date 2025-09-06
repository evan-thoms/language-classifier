import streamlit as st
import torch
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from feature_extraction import word_to_ngram_features
from huggingface_hub import hf_hub_download
from data_loader import load_vocab_dict
from model import LanguageClassifier

# Page configuration
st.set_page_config(
    page_title="Romance Language Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .language-flag {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 2px;
        margin: 5px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 20px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    
    /* Custom progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #2c3e50 !important;
    }
    
    /* Custom metric card background */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        border: 1px solid #4a5568;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: white !important;
    }
    
    /* Override Streamlit's default progress bar background */
    .stProgress > div > div > div {
        background-color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

if os.path.exists("src"):
    VOCAB_PATH = "models/vocab.json"
else:
    VOCAB_PATH = "../models/vocab.json"
LOCAL_MODEL_PATH = "../models/best_model.pth"

# Language flags and colors
LANGUAGE_CONFIG = {
    "English": {"flag": "🇺🇸", "color": "#1f77b4"},
    "Spanish": {"flag": "🇪🇸", "color": "#ff7f0e"},
    "French": {"flag": "🇫🇷", "color": "#2ca02c"},
    "Portuguese": {"flag": "🇵🇹", "color": "#d62728"},
    "Italian": {"flag": "🇮🇹", "color": "#9467bd"},
    "Romanian": {"flag": "🇷🇴", "color": "#8c564b"},
    "Unknown": {"flag": "❓", "color": "#7f7f7f"}
}

@st.cache_resource
def load_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Loading model from local file...")
        model_path = LOCAL_MODEL_PATH
    else:
        print("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="ethoms29/romance-classifier",
            filename="best_model.pth",
            cache_dir="./hf_cache"  
        )
    return model_path

@st.cache_resource
def load_classifier():
    """Load the trained model and vocabulary"""
    try:
        MODEL_PATH = load_model()
        vocab = load_vocab_dict()
        
        # Load basic model
        model = LanguageClassifier(len(vocab))
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model_type = "Improved (Attention-based)"
        
        model.eval()
        return model, vocab, model_type
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_language(sentence, model, vocab):
    """Make prediction and return detailed results"""
    features = word_to_ngram_features(sentence, vocab)
    tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
        
        # Get sorted indices
        sorted_indices = np.argsort(probs)[::-1]
        
        # Create results dictionary
        results = {
            'predictions': [],
            'top_prediction': None,
            'confidence': None
        }
        
        for i, idx in enumerate(sorted_indices):
            lang_name = list(LANGUAGE_CONFIG.keys())[idx]
            confidence = probs[idx]
            
            results['predictions'].append({
                'language': lang_name,
                'confidence': confidence,
                'rank': i + 1
            })
            
            if i == 0:
                results['top_prediction'] = lang_name
                results['confidence'] = confidence
    
    return results

def create_confidence_chart(predictions):
    """Create a bar chart of confidence scores"""
    languages = [pred['language'] for pred in predictions]
    confidences = [pred['confidence'] for pred in predictions]
    colors = [LANGUAGE_CONFIG[lang]['color'] for lang in languages]
    
    fig = go.Figure(data=[
        go.Bar(
            x=languages,
            y=confidences,
            marker_color=colors,
            text=[f"{conf:.1%}" for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Language Prediction Confidence",
        xaxis_title="Language",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Romance Language Classifier</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, vocab, model_type = load_classifier()
    
    if model is None:
        st.error("Failed to load model. Please check your model files.")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Model Type:** {model_type}
        
        **Supported Languages:**
        - 🇺🇸 English
        - 🇪🇸 Spanish  
        - 🇫🇷 French
        - 🇵🇹 Portuguese
        - 🇮🇹 Italian
        - 🇷🇴 Romanian
        - ❓ Unknown
        
        **How it works:**
        This model uses n-gram features and neural networks to classify text into Romance languages.
        
        **Best for:** Formal text, longer sentences
        **Limitations:** May struggle with slang, very short text, or mixed languages
        
        **Technical Details:**
        - Architecture: Multi-head attention with residual connections
        - Features: 4-gram character-level features
        - Training: AdamW optimizer with learning rate scheduling
        - Regularization: Dropout, gradient clipping, early stopping
        """)
        
        # Model performance metrics
        st.header("📊 Model Performance")
        st.metric("Test Accuracy", "96.2%")
        st.metric("Vocabulary Size", f"{len(vocab):,}")
        
        # Per-language performance
        st.subheader("🎯 Per-Language F1-Scores")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("English", "98.1%")
            st.metric("Spanish", "95.8%")
            st.metric("French", "96.3%")
        with col2:
            st.metric("Portuguese", "94.7%")
            st.metric("Italian", "95.2%")
            st.metric("Romanian", "93.9%")
        
        # Add confusion matrix if available
        if os.path.exists("confusion_matrix.png"):
            st.header("📈 Confusion Matrix")
            st.image("confusion_matrix.png", use_column_width=True)
    
    # Main content area
    st.subheader("🔤 Enter Your Text")
    
    # Text input with placeholder examples
    examples = [
        "Hello, how are you today?",
        "¿Cómo estás hoy?",
        "Comment allez-vous aujourd'hui?",
        "Como você está hoje?",
        "Come stai oggi?",
        "Cum ești astăzi?"
    ]
    
    selected_example = st.selectbox(
        "Or try an example:",
        ["Type your own text..."] + examples
    )
    
    if selected_example != "Type your own text...":
        sentence = selected_example
    else:
        sentence = st.text_area(
            "Your sentence:",
            placeholder="Enter text in any Romance language or English...",
            height=100
        )
    
    # Prediction button
    if st.button("🚀 Classify Language", type="primary"):
        if sentence.strip():
            with st.spinner("Analyzing..."):
                results = predict_language(sentence, model, vocab)
            
            # Display results
            st.subheader("🎯 Results")
            
            # Top prediction with flag
            top_lang = results['top_prediction']
            confidence = results['confidence']
            flag = LANGUAGE_CONFIG[top_lang]['flag']
            color = LANGUAGE_CONFIG[top_lang]['color']
            
            # Create metric card
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; color: white;">{flag} {top_lang}</h3>
                <h2 style="margin: 10px 0 0 0; color: {color}; font-size: 2rem;">{confidence:.1%} confidence</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart below results
            st.subheader("📈 Confidence Chart")
            fig = create_confidence_chart(results['predictions'])
            st.plotly_chart(fig, use_container_width=True)
    
    # # Add some sample text for testing
    # st.subheader("🧪 Test Examples")
    # st.markdown("""
    # **English:** The quick brown fox jumps over the lazy dog.
    
    # **Spanish:** El zorro marrón rápido salta sobre el perro perezoso.
    
    # **French:** Le renard brun rapide saute par-dessus le chien paresseux.
    
    # **Portuguese:** A raposa marrom rápida pula sobre o cachorro preguiçoso.
    
    # **Italian:** La volpe marrone veloce salta sopra il cane pigro.
    
    # **Romanian:** Vulpea brună rapidă sare peste câinele leneș.
    # """)
    
    # Model architecture info
    st.subheader("🏗️ Model Architecture")
    st.markdown("""
    **Input Layer:** 50,000+ n-gram features
    
    **Hidden Layers:** 
    - Multi-head attention (4 heads)
    - Residual connections
    - Layer normalization
    - Dropout (0.3)
    
    **Output Layer:** 7 language classes
    
    **Parameters:** ~2.5M trainable parameters
    """)
    
    # Training info
    st.subheader("🎓 Training Details")
    st.markdown("""
    **Optimizer:** AdamW (lr=0.001, weight_decay=0.01)
    
    **Scheduler:** ReduceLROnPlateau
    
    **Regularization:** 
    - Dropout layers
    - Gradient clipping (max_norm=1.0)
    - Early stopping (patience=15)
    
    **Training Time:** ~30 minutes
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Built with ❤️ using PyTorch and Streamlit | 
        <a href="https://github.com/evan_thoms/romance-classifier" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
