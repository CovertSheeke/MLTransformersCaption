import streamlit as st
import os
from PIL import Image
import io
from pathlib import Path

# Import your inference functions
try:
    from model_self_attention import VisionLanguageEncoder, CaptionDecoder
    import sys
    sys.path.append('.')
    
    # Import from the refactored file (note: filename starts with 12_)
    import importlib.util
    spec = importlib.util.spec_from_file_location("single_image_inference", "12_single_image_inference.py")
    single_image_inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(single_image_inference)
    
    load_model = single_image_inference.load_model
    generate_caption_for_image = single_image_inference.generate_caption_for_image
    MODEL_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import model components: {e}")
    MODEL_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'caption_generated' not in st.session_state:
    st.session_state.caption_generated = False

@st.cache_resource
def initialize_models():
    """Load models once and cache them"""
    if not MODEL_AVAILABLE:
        return False
    
    try:
        # Check if model file exists
        model_path = 'saved_models/self_attention/model_checkpoint.pth'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return False
        
        load_model(model_path)
        return True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False

def main():
    # App title
    st.markdown('<div class="main-header">🖼️ Image Caption Generator</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model parameters
        max_tokens = st.slider("Max Tokens", min_value=10, max_value=100, value=30, step=5)
        top_k = st.slider("Top-k Sampling", min_value=10, max_value=100, value=50, step=10)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses a vision-language model to generate captions for uploaded images.
        
        **Instructions:**
        1. Upload an image using the file uploader
        2. Adjust parameters if needed
        3. Click 'Generate Caption' to get results
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="Supported formats: JPG, JPEG, PNG, BMP, GIF"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generate caption button
            if st.button("🚀 Generate Caption", type="primary"):
                if not MODEL_AVAILABLE:
                    st.error("Model components are not available. Please check the installation.")
                    return
                
                with st.spinner("Loading models..."):
                    models_loaded = initialize_models()
                
                if not models_loaded:
                    st.error("Failed to load models. Please check the model files.")
                    return
                
                with st.spinner("Generating caption..."):
                    try:
                        # Generate caption
                        predicted_caption, fig = generate_caption_for_image(
                            image,  # Pass PIL image directly
                            max_tokens=max_tokens,
                            top_k=top_k
                        )
                        
                        # Store results in session state
                        st.session_state.predicted_caption = predicted_caption
                        st.session_state.generated_figure = fig
                        st.session_state.caption_generated = True
                        
                        st.success("Caption generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating caption: {e}")
                        st.exception(e)
    
    with col2:
        st.header("📝 Generated Caption")
        
        if st.session_state.caption_generated:
            # Display the generated caption
            st.markdown(f"""
            <div class="caption-box">
                <h3>Generated Caption:</h3>
                <p style="font-size: 1.2rem; font-weight: bold; color: #333;">
                    "{st.session_state.predicted_caption}"
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the matplotlib figure
            if hasattr(st.session_state, 'generated_figure'):
                st.pyplot(st.session_state.generated_figure)
            
            # Download options
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            # Create download data
            caption_text = f"Generated Caption: {st.session_state.predicted_caption}"
            
            # Download caption as text
            st.download_button(
                label="📄 Download Caption (TXT)",
                data=caption_text,
                file_name="generated_caption.txt",
                mime="text/plain"
            )
            
        else:
            st.info("👆 Upload an image and click 'Generate Caption' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "Built with ❤️ using Streamlit and PyTorch"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
