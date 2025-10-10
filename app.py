"""
Trymylook Virtual Makeup Try-On
Main Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Import custom modules
from config import (
    APP_TITLE, APP_ICON, VERSION, 
    PRODUCTS, get_shades_for_product, 
    DEFAULT_INTENSITY, MIN_INTENSITY, MAX_INTENSITY,
    MAX_IMAGE_SIZE, MIN_IMAGE_SIZE
)
from face_detection import FaceDetector
from segmentation import MakeupSegmenter
from makeup_application import MakeupApplicator
from utils import (
    pil_to_cv, cv_to_pil, resize_image, ensure_min_size,
    create_side_by_side, validate_image, draw_detection_boxes
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'face_data' not in st.session_state:
    st.session_state.face_data = None
if 'mask' not in st.session_state:
    st.session_state.mask = None

# ============================================================================
# INITIALIZE MODULES
# ============================================================================

@st.cache_resource
def load_models():
    """Load face detection and processing models"""
    detector = FaceDetector()
    segmenter = MakeupSegmenter()
    applicator = MakeupApplicator()
    return detector, segmenter, applicator

detector, segmenter, applicator = load_models()

# ============================================================================
# SIDEBAR - CONTROLS
# ============================================================================

st.sidebar.title(f"{APP_ICON} Makeup Controls")
st.sidebar.markdown(f"**Version:** {VERSION}")
st.sidebar.markdown("---")

# Product selection
product = st.sidebar.selectbox(
    "📦 Select Product",
    PRODUCTS,
    help="Choose the makeup product to apply"
)

# Get shades for selected product
shades = get_shades_for_product(product)
shade_names = list(shades.keys())

# Shade selection
selected_shade = st.sidebar.selectbox(
    "🎨 Select Shade",
    shade_names,
    help="Choose the color shade"
)

# Display color swatch
shade_rgb = shades[selected_shade]
swatch_html = f"""
<div style="
    background-color: rgb{shade_rgb};
    width: 100%;
    height: 50px;
    border-radius: 8px;
    border: 2px solid #ddd;
    margin: 10px 0;
"></div>
"""
st.sidebar.markdown(swatch_html, unsafe_allow_html=True)

# Intensity slider
intensity = st.sidebar.slider(
    "💪 Intensity",
    min_value=MIN_INTENSITY,
    max_value=MAX_INTENSITY,
    value=DEFAULT_INTENSITY,
    help="Adjust makeup intensity (0 = subtle, 100 = bold)"
)

st.sidebar.markdown("---")

# Additional options
show_comparison = st.sidebar.checkbox("👁️ Show Before/After", value=True)
show_mask = st.sidebar.checkbox("🎭 Show Detection Mask", value=False)
show_detections = st.sidebar.checkbox("📦 Show Detection Boxes", value=False)

st.sidebar.markdown("---")

# Reset button
if st.sidebar.button("🔄 Reset All", use_container_width=True):
    st.session_state.processed_image = None
    st.session_state.original_image = None
    st.session_state.face_data = None
    st.session_state.mask = None
    st.rerun()

# ============================================================================
# MAIN AREA - HEADER
# ============================================================================

st.title(APP_TITLE)
st.markdown("""
Upload a selfie and apply virtual makeup with adjustable intensity.  
Try different products, shades, and settings using the controls on the left! 💄✨
""")

st.markdown("---")

# ============================================================================
# MAIN AREA - IMAGE UPLOAD
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Your Selfie")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a selfie...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear, front-facing photo for best results"
    )
    
    # Camera input option
    use_camera = st.checkbox("📷 Use Camera Instead")
    
    if use_camera:
        camera_image = st.camera_input("Take a selfie")
        if camera_image:
            uploaded_file = camera_image
    
    # Process uploaded image
    if uploaded_file is not None:
        try:
            # Load image
            pil_image = Image.open(uploaded_file)
            cv_image = pil_to_cv(pil_image)
            
            # Validate image
            is_valid, message = validate_image(cv_image)
            
            if not is_valid:
                st.error(f"❌ {message}")
                st.stop()
            
            # Resize if needed
            cv_image = resize_image(cv_image, MAX_IMAGE_SIZE)
            cv_image = ensure_min_size(cv_image, MIN_IMAGE_SIZE)
            
            if cv_image is None:
                st.error("❌ Image is too small to process")
                st.stop()
            
            # Store original image
            st.session_state.original_image = cv_image
            
            # Display original image
            st.image(cv_to_pil(cv_image), caption="Original Image", use_container_width=True)
            
            # Image info
            h, w = cv_image.shape[:2]
            st.info(f"📏 Image size: {w} x {h} pixels")
            
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.stop()
    else:
        st.info("👆 Upload a selfie or use camera to get started")

with col2:
    st.subheader("✨ Result")
    
    if st.session_state.original_image is not None:
        # Apply Makeup button
        if st.button("🎨 Apply Makeup", type="primary", use_container_width=True):
            with st.spinner("🔮 Detecting face and applying makeup..."):
                try:
                    # Step 1: Detect face
                    face_data = detector.detect_all(st.session_state.original_image)
                    
                    if face_data is None:
                        st.error("❌ No face detected! Please try a different photo with a clear, front-facing face.")
                        st.stop()
                    
                    st.session_state.face_data = face_data
                    
                    # Step 2: Create mask
                    mask = segmenter.create_mask_for_product(
                        st.session_state.original_image.shape,
                        product,
                        face_data
                    )
                    
                    st.session_state.mask = mask
                    
                    # Step 3: Apply makeup
                    if product == "Lipstick":
                        result = applicator.apply_lipstick(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    elif product == "Eyeshadow":
                        result = applicator.apply_eyeshadow(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    elif product == "Foundation":
                        result = applicator.apply_foundation(
                            st.session_state.original_image,
                            mask,
                            shade_rgb,
                            intensity
                        )
                    else:
                        result = st.session_state.original_image
                    
                    st.session_state.processed_image = result
                    st.success("✅ Makeup applied successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.stop()
        
        # Display result
        if st.session_state.processed_image is not None:
            if show_comparison:
                # Show before/after
                comparison = create_side_by_side(
                    st.session_state.original_image,
                    st.session_state.processed_image,
                    labels=("Before", "After")
                )
                st.image(cv_to_pil(comparison), caption="Before & After", use_container_width=True)
            else:
                # Show only result
                st.image(
                    cv_to_pil(st.session_state.processed_image),
                    caption=f"{product} Applied - {selected_shade}",
                    use_container_width=True
                )
            
            # Download button
            result_pil = cv_to_pil(st.session_state.processed_image)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            
            st.download_button(
                label="📥 Download Result",
                data=buf.getvalue(),
                file_name=f"trymylook_{product.lower()}_{selected_shade.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("👆 Click 'Apply Makeup' to see the result!")
    else:
        st.info("📸 Upload an image first to see results here")

# ============================================================================
# DEBUG VISUALIZATIONS
# ============================================================================

if show_mask and st.session_state.mask is not None:
    st.markdown("---")
    st.subheader("🎭 Detection Mask Visualization")
    
    # Convert mask to colored visualization
    mask_colored = cv2.applyColorMap(st.session_state.mask, cv2.COLORMAP_JET)
    mask_overlay = cv2.addWeighted(
        st.session_state.original_image, 0.6,
        mask_colored, 0.4, 0
    )
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.image(cv_to_pil(st.session_state.mask), caption="Binary Mask", use_container_width=True)
    
    with col_m2:
        st.image(cv_to_pil(mask_colored), caption="Heatmap", use_container_width=True)
    
    with col_m3:
        st.image(cv_to_pil(mask_overlay), caption="Overlay", use_container_width=True)

if show_detections and st.session_state.face_data is not None:
    st.markdown("---")
    st.subheader("📦 Face Detection Visualization")
    
    detected_viz = draw_detection_boxes(
        st.session_state.original_image,
        st.session_state.face_data
    )
    
    st.image(cv_to_pil(detected_viz), caption="Detected Regions", use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>💄 <strong>Trymylook Virtual Makeup</strong> • Built with Streamlit & OpenCV</p>
    <p>For best results: Use well-lit, front-facing photos • Resolution: 640x480 to 1920x1080</p>
</div>
""", unsafe_allow_html=True)