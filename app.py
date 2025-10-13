"""
Trymylook Virtual Makeup Try-On
Main Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

from config import (
    APP_TITLE, APP_ICON, VERSION, 
    PRODUCTS, get_shades_for_product, 
    DEFAULT_INTENSITY, MIN_INTENSITY, MAX_INTENSITY,
    MAX_IMAGE_SIZE, MIN_IMAGE_SIZE
)
from face_detection_dl import DeepLearningFaceDetector
from segmentation_dl import NeuralSegmenter
from makeup_application_dl import NeuralMakeupApplicator
from utils import (
    pil_to_cv, cv_to_pil, resize_image, ensure_min_size,
    create_side_by_side, validate_image, get_image_info
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'landmarks' not in st.session_state:
    st.session_state.landmarks = None
if 'face_data' not in st.session_state:
    st.session_state.face_data = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None


@st.cache_resource
def load_models():
    with st.spinner("🔮 Loading deep learning models..."):
        detector = DeepLearningFaceDetector(device='cpu')
        segmenter = NeuralSegmenter()
        applicator = NeuralMakeupApplicator()
    return detector, segmenter, applicator


try:
    detector, segmenter, applicator = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()


st.sidebar.title(f"{APP_ICON} Makeup Controls")
st.sidebar.markdown(f"**Version:** {VERSION}")
st.sidebar.markdown("---")

product = st.sidebar.selectbox(
    "📦 Select Product",
    PRODUCTS,
    help="Choose the makeup product to apply"
)

shades = get_shades_for_product(product)
shade_names = list(shades.keys())

selected_shade = st.sidebar.selectbox(
    "🎨 Select Shade",
    shade_names,
    help="Choose the color shade"
)

shade_rgb = shades[selected_shade]
swatch_html = f"""
<div style="
    background-color: rgb{shade_rgb};
    width: 100%;
    height: 50px;
    border-radius: 8px;
    border: 2px solid #ddd;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
"></div>
<p style="text-align: center; color: #666; font-size: 0.9em;">{selected_shade}</p>
"""
st.sidebar.markdown(swatch_html, unsafe_allow_html=True)

intensity = st.sidebar.slider(
    "💪 Intensity",
    min_value=MIN_INTENSITY,
    max_value=MAX_INTENSITY,
    value=DEFAULT_INTENSITY,
    help="Adjust makeup intensity (0 = subtle, 100 = bold)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Options")

show_comparison = st.sidebar.checkbox("👁️ Show Before/After", value=True)
show_processing_time = st.sidebar.checkbox("⏱️ Show Processing Time", value=True)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset All", use_container_width=True):
    st.session_state.processed_image = None
    st.session_state.original_image = None
    st.session_state.landmarks = None
    st.session_state.face_data = None
    st.session_state.processing_time = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Tips for Best Results
- Use well-lit photos
- Front-facing angle (< 30°)
- Clear, unobstructed face
- Resolution: 640x480 to 1920x1080

### 🎨 Intensity Guide
- **20-40%**: Natural, subtle
- **50-70%**: Moderate
- **80-100%**: Bold, dramatic
""")


st.title(APP_TITLE)
st.markdown(f"""
Upload a selfie and apply virtual makeup with adjustable intensity using deep learning.  
Powered by Face-Alignment Network (97% accuracy) • {len(shade_names)} shades available
""")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Your Selfie")
    
    uploaded_file = st.file_uploader(
        "Choose a selfie...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear, front-facing photo for best results"
    )
    
    use_camera = st.checkbox("📷 Use Camera Instead")
    
    if use_camera:
        camera_image = st.camera_input("Take a selfie")
        if camera_image:
            uploaded_file = camera_image
    
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            cv_image = pil_to_cv(pil_image)
            
            is_valid, message = validate_image(cv_image)
            
            if not is_valid:
                st.error(f"❌ {message}")
                st.stop()
            
            cv_image = resize_image(cv_image, MAX_IMAGE_SIZE)
            cv_image = ensure_min_size(cv_image, MIN_IMAGE_SIZE)
            
            if cv_image is None:
                st.error("❌ Image is too small to process. Minimum size: 400x400 pixels")
                st.stop()
            
            st.session_state.original_image = cv_image
            
            st.image(cv_to_pil(cv_image), caption="Original Image", use_container_width=True)
            
            img_info = get_image_info(cv_image)
            st.info(f"📏 Image size: {img_info['width']} x {img_info['height']} pixels")
            
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.stop()
    else:
        st.info("👆 Upload a selfie or use camera to get started")
        
        st.markdown("""
        ### 📋 Supported Formats
        - JPG / JPEG
        - PNG
        
        ### ✅ Requirements
        - Clear, well-lit photo
        - Front-facing or slight angle
        - Single face visible
        - Min size: 400x400 pixels
        - Max size: 4000x4000 pixels
        """)

with col2:
    st.subheader("✨ Result")
    
    if st.session_state.original_image is not None:
        
        apply_button = st.button(
            "🎨 Apply Makeup", 
            type="primary", 
            use_container_width=True,
            help=f"Apply {product} - {selected_shade} at {intensity}% intensity"
        )
        
        if apply_button:
            with st.spinner("🔮 Detecting face and applying makeup..."):
                start_time = time.time()
                
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Step 1/3: Detecting face landmarks...")
                    progress_bar.progress(10)
                    
                    result = detector.detect_all(st.session_state.original_image)
                    
                    if result is None:
                        st.error("❌ No face detected! Please try:")
                        st.markdown("""
                        - Use a well-lit photo
                        - Ensure face is clearly visible
                        - Try a front-facing angle
                        - Remove obstructions (hair, hands)
                        """)
                        st.stop()
                    
                    landmarks = result['landmarks']
                    st.session_state.landmarks = landmarks
                    st.session_state.face_data = result
                    
                    progress_bar.progress(33)
                    status_text.text("Step 2/3: Creating segmentation mask...")
                    
                    mask = segmenter.create_mask_for_product(
                        st.session_state.original_image.shape,
                        product,
                        landmarks
                    )
                    
                    progress_bar.progress(66)
                    status_text.text(f"Step 3/3: Applying {product}...")
                    
                    if product == "Lipstick":
                        processed = applicator.apply_lipstick(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    elif product == "Eyeshadow":
                        processed = applicator.apply_eyeshadow(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    elif product == "Foundation":
                        processed = applicator.apply_foundation(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    elif product == "Blush":
                        processed = applicator.apply_blush(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    else:
                        processed = st.session_state.original_image
                    
                    st.session_state.processed_image = processed
                    
                    end_time = time.time()
                    st.session_state.processing_time = end_time - start_time
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Makeup applied successfully! ({st.session_state.processing_time:.2f}s)")
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        if st.session_state.processed_image is not None:
            if show_comparison:
                comparison = create_side_by_side(
                    st.session_state.original_image,
                    st.session_state.processed_image,
                    labels=("Before", "After")
                )
                st.image(cv_to_pil(comparison), caption="Before & After Comparison", use_container_width=True)
            else:
                st.image(
                    cv_to_pil(st.session_state.processed_image),
                    caption=f"{product} Applied - {selected_shade} ({intensity}%)",
                    use_container_width=True
                )
            
            if show_processing_time and st.session_state.processing_time:
                st.metric(
                    label="Processing Time",
                    value=f"{st.session_state.processing_time:.2f}s",
                    delta="Deep Learning Pipeline"
                )
            
            result_pil = cv_to_pil(st.session_state.processed_image)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name=f"trymylook_{product.lower()}_{selected_shade.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_d2:
                if st.button("🔄 Try Different Settings", use_container_width=True):
                    st.session_state.processed_image = None
                    st.rerun()
            
            if st.session_state.face_data:
                with st.expander("🔍 Detection Details"):
                    st.json({
                        "Landmarks Detected": len(st.session_state.landmarks),
                        "Face Center": st.session_state.face_data.get('center'),
                        "Face Angle": f"{st.session_state.face_data.get('angle', 0):.2f}°",
                        "Confidence": st.session_state.face_data.get('confidence', 1.0),
                        "Product": product,
                        "Shade": selected_shade,
                        "Intensity": f"{intensity}%"
                    })
        
        else:
            st.info("👆 Click 'Apply Makeup' to see the result!")
            
            st.markdown("""
            ### 🎨 What happens next?
            1. **Face Detection**: AI finds your face (97% accuracy)
            2. **Landmark Detection**: 68 precise points mapped
            3. **Segmentation**: Perfect masks for lips/eyes/skin
            4. **Makeup Application**: Realistic blending with texture preservation
            5. **Result**: Professional-quality virtual makeup!
            
            Processing takes 2-3 seconds on CPU, <1 second on GPU.
            """)
    
    else:
        st.info("📸 Upload an image first to see results here")
        
        st.markdown("""
        ### 🌟 Features
        - **Deep Learning Face Detection** (97% accuracy)
        - **68 Facial Landmarks** for precision
        - **3 Products**: Lipstick, Eyeshadow, Foundation
        - **19 Professional Shades**
        - **Adjustable Intensity** (0-100%)
        - **Before/After Comparison**
        - **Download Results** in high quality
        
        ### 🚀 Powered By
        - Face-Alignment Network (PyTorch)
        - ResNet-50 + Hourglass Architecture
        - Advanced Neural Blending
        """)


st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p style="font-size: 1.1em;"><strong>💄 Trymylook Virtual Makeup Try-On</strong></p>
    <p>Powered by Deep Learning • Face-Alignment Network (97% Accuracy)</p>
    <p style="font-size: 0.9em;">Built with Streamlit, PyTorch & OpenCV • Version {}</p>
    <p style="font-size: 0.85em; margin-top: 10px;">
        For best results: Use well-lit, front-facing photos • Resolution: 640x480 to 1920x1080
    </p>
</div>
""".format(VERSION), unsafe_allow_html=True)


with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Products", len(PRODUCTS))
    st.metric("Total Shades", len(shade_names))
    if st.session_state.processing_time:
        st.metric("Last Processing", f"{st.session_state.processing_time:.2f}s")