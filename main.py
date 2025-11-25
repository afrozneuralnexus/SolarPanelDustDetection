import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from PIL import Image
import json
import os

# Page configuration
st.set_page_config(
    page_title="Solar Panel Dust Detection",
    page_icon="☀️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF6B35;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #6C757D;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .dusty {
        background-color: #FFE5E5;
        border: 2px solid #FF6B6B;
    }
    .clean {
        background-color: #E5F9E5;
        border: 2px solid #51CF66;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    try:
        # Get current directory and list files
        current_dir = os.getcwd()
        files_in_dir = os.listdir(current_dir)
        
        # Debug information
        st.info(f"📂 Current directory: {current_dir}")
        st.info(f"📄 Files found: {', '.join(files_in_dir)}")
        
        # Try multiple possible paths
        possible_paths = [
            ('inception_dust_model.h5', 'labels.json'),
            (os.path.join(current_dir, 'inception_dust_model.h5'), os.path.join(current_dir, 'labels.json')),
        ]
        
        model_path = None
        labels_path = None
        
        for mp, lp in possible_paths:
            if os.path.exists(mp) and os.path.exists(lp):
                model_path = mp
                labels_path = lp
                break
        
        if model_path is None:
            st.error("❌ Model file 'inception_dust_model.h5' not found")
            st.warning("Please ensure the file is uploaded to the root of your GitHub repository")
            return None, None
            
        if labels_path is None:
            st.error("❌ Labels file 'labels.json' not found")
            st.warning("Please ensure the file is uploaded to the root of your GitHub repository")
            return None, None
        
        st.success(f"✅ Found model at: {model_path}")
        st.success(f"✅ Found labels at: {labels_path}")
        
        model = load_model(model_path)
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        st.success("✅ Model and labels loaded successfully!")
        return model, labels
    except Exception as e:
        st.error(f"❌ Error loading model or labels: {str(e)}")
        st.exception(e)
        return None, None

# Prediction function
def predict_image(img, model, labels):
    try:
        # Convert to RGB if necessary (handles PNG with alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to model input size
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction with verbosity off
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = labels[str(predicted_class_idx)]
        confidence = predictions[0][predicted_class_idx] * 100
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        raise e

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">☀️ Solar Panel Dust Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of a solar panel to check if it\'s dusty or clean</p>', unsafe_allow_html=True)
    
    # Load model
    model, labels = load_model_and_labels()
    
    if model is None or labels is None:
        st.error("⚠️ Please ensure 'inception_dust_model.h5' and 'labels.json' are in the same directory as this script.")
        return
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a solar panel image (JPG, JPEG, or PNG format)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_container_width=True)
        
        # Predict button
        if st.button('🔍 Analyze Image', use_container_width=True):
            with st.spinner('Analyzing...'):
                try:
                    predicted_class, confidence, all_predictions = predict_image(img, model, labels)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    # Result box with conditional styling
                    if predicted_class.lower() == 'dusty':
                        st.markdown(f"""
                            <div class="result-box dusty">
                                <h2>🚨 Dusty Solar Panel</h2>
                                <p style="font-size: 1.5em; font-weight: bold;">Confidence: {confidence:.2f}%</p>
                                <p>⚠️ Cleaning recommended to maintain optimal efficiency</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="result-box clean">
                                <h2>✅ Clean Solar Panel</h2>
                                <p style="font-size: 1.5em; font-weight: bold;">Confidence: {confidence:.2f}%</p>
                                <p>👍 Panel is in good condition</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed predictions
                    with st.expander("View Detailed Predictions"):
                        for idx, prob in enumerate(all_predictions):
                            class_name = labels[str(idx)]
                            st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
                            
                except Exception as e:
                    st.error("❌ An error occurred during prediction. Please try with a different image.")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6C757D; padding: 20px;">
            <p>💡 <strong>Tip:</strong> For best results, use clear images of solar panels</p>
            <p style="font-size: 0.9em;">Powered by InceptionV3 Transfer Learning</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
