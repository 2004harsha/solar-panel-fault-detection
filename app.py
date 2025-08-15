import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Solar Panel Fault Detection ☀️",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model with caching for better performance
@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    try:
        model = tf.keras.models.load_model('models/solar_panel_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image preprocessing function
def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Adjust size based on your model's input requirements
    img = image.resize((224, 224))  
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def main():
    st.title("🌞 Solar Panel Fault Detection System")
    st.write("Upload an image of a solar panel to detect faults automatically using AI")
    
    # Sidebar with controls
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
        
        st.header("Model Information")
        st.info("""
        This CNN model detects:
        - Normal panels ✅
        - Cracked panels ⚠️
        - Dusty/dirty panels 🧹
        - Hot spots 🔥
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a solar panel"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            model = load_model()
            if model is not None:
                with st.spinner('Analyzing image...'):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    
                    if prediction is not None:
                        # Update these labels to match YOUR model's classes
                        class_labels = ['Normal', 'Cracked', 'Dusty', 'Hot Spot']
                        predicted_class_idx = np.argmax(prediction[0])
                        predicted_class = class_labels[predicted_class_idx]
                        confidence = float(prediction[predicted_class_idx])
                        
                        # Display results
                        if confidence > confidence_threshold:
                            if predicted_class == 'Normal':
                                st.success(f"✅ **{predicted_class}** panel detected")
                            else:
                                st.error(f"⚠️ **{predicted_class}** fault detected!")
                            st.write(f"Confidence: {confidence:.2%}")
                        else:
                            st.warning("Low confidence. Try a clearer image.")
                        
                        # Show all predictions
                        st.subheader("Detailed Predictions:")
                        for label, prob in zip(class_labels, prediction[0]):
                            st.write(f"{label}: {prob:.2%}")
                            st.progress(float(prob))
    
    else:
        st.info("👆 Please upload an image to get started")

if __name__ == "__main__":
    main()
