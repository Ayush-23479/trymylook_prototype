"""
Trymylook Virtual Makeup Try-On
<<<<<<< HEAD
Main Streamlit Application - WITH BISENET INTEGRATION
=======
Main Streamlit Application with Complete Look Feature
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
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
from segmentation_bisenet import HybridSegmenter
 # ⭐ CHANGED: Using BiSeNet
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
if 'parsing_map' not in st.session_state:  # ⭐ NEW: Store BiSeNet parsing
    st.session_state.parsing_map = None


@st.cache_resource
def load_models():
    with st.spinner("🔮 Loading deep learning models (including BiSeNet)..."):
        detector = DeepLearningFaceDetector(device='cpu')
        segmenter = HybridSegmenter(device='cpu')  # ⭐ CHANGED: BiSeNet segmenter
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

if product == "Complete Look":
    selected_preset = st.sidebar.selectbox(
        "🎨 Select Preset Look",
        shade_names,
        help="Choose a complete makeup look"
    )
    
    preset_details = shades[selected_preset]
    
    st.sidebar.markdown("### 📋 Preset Details:")
    for makeup_type, (shade, intensity_val) in preset_details.items():
        st.sidebar.markdown(f"**{makeup_type.title()}:** {shade} ({intensity_val}%)")
    
    st.sidebar.markdown("---")
    
    selected_shade = selected_preset
    shade_rgb = (255, 192, 203)
    
else:
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

if product != "Complete Look":
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
show_parsing_viz = st.sidebar.checkbox("🔬 Show BiSeNet Parsing", value=False)  # ⭐ NEW

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset All", use_container_width=True):
    st.session_state.processed_image = None
    st.session_state.original_image = None
    st.session_state.landmarks = None
    st.session_state.face_data = None
    st.session_state.processing_time = None
    st.session_state.parsing_map = None
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

### 🔬 BiSeNet Features
- **19-class** semantic segmentation
- **Pixel-perfect** face parsing
- **Professional** quality masks
""")


st.title(APP_TITLE)
st.markdown(f"""
Upload a selfie and apply virtual makeup with adjustable intensity using deep learning.  
<<<<<<< HEAD
Powered by **BiSeNet Face Parsing** + Face-Alignment Network • {len(shade_names)} shades available
=======
Powered by Face-Alignment Network (97% accuracy) • 5 products • 25+ shades • 5 complete looks
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
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
        
        if product == "Complete Look":
            button_text = f"🎨 Apply Complete Look: {selected_preset}"
        else:
            button_text = f"🎨 Apply {product}"
        
        apply_button = st.button(
            button_text, 
            type="primary", 
            use_container_width=True
        )
        
        if apply_button:
            with st.spinner("🔮 Processing with BiSeNet face parsing..."):
                start_time = time.time()
                
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # ⭐ CHANGED: Using BiSeNet for segmentation
                    status_text.text("Step 1/4: Detecting face landmarks...")
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
                    
<<<<<<< HEAD
                    progress_bar.progress(25)
                    status_text.text("Step 2/4: Running BiSeNet face parsing...")
                    
                    # ⭐ CHANGED: Create mask using BiSeNet with the actual image
                    mask = segmenter.create_mask_for_product(
                        st.session_state.original_image,  # Pass image, not shape
                        product,
                        landmarks  # Fallback if BiSeNet fails
                    )
                    
                    progress_bar.progress(50)
                    status_text.text("Step 3/4: Refining segmentation mask...")
                    
                    # Store for visualization
                    if hasattr(segmenter, 'bisenet') and segmenter.use_bisenet:
                        st.session_state.parsing_map = segmenter.bisenet.segment(
                            st.session_state.original_image
                        )
                    
                    progress_bar.progress(75)
                    status_text.text(f"Step 4/4: Applying {product}...")
                    
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
=======
                    progress_bar.progress(33)
                    status_text.text("Step 2/3: Creating segmentation masks...")
                    
                    if product == "Complete Look":
                        preset = get_shades_for_product("Complete Look")[selected_preset]
                        processed = st.session_state.original_image.copy()
                        
                        progress_bar.progress(50)
                        
                        steps = [
                            ("Foundation", "foundation"),
                            ("Blush", "blush"),
                            ("Eyeshadow", "eyeshadow"),
                            ("Lipstick", "lipstick")
                        ]
                        
                        for idx, (makeup_name, makeup_key) in enumerate(steps):
                            if makeup_key in preset:
                                shade_name, makeup_intensity = preset[makeup_key]
                                
                                step_progress = 50 + int((idx + 1) / len(steps) * 45)
                                progress_bar.progress(step_progress)
                                status_text.text(f"Step 3/3: Applying {makeup_name}... ({idx+1}/{len(steps)})")
                                
                                makeup_shades = get_shades_for_product(makeup_name)
                                if shade_name in makeup_shades:
                                    makeup_color = makeup_shades[shade_name]
                                    
                                    makeup_mask = segmenter.create_mask_for_product(
                                        processed.shape,
                                        makeup_name,
                                        landmarks
                                    )
                                    
                                    if makeup_name == "Lipstick":
                                        processed = applicator.apply_lipstick(
                                            processed, makeup_mask, makeup_color, makeup_intensity
                                        )
                                    elif makeup_name == "Eyeshadow":
                                        processed = applicator.apply_eyeshadow(
                                            processed, makeup_mask, makeup_color, makeup_intensity
                                        )
                                    elif makeup_name == "Foundation":
                                        processed = applicator.apply_foundation(
                                            processed, makeup_mask, makeup_color, makeup_intensity
                                        )
                                    elif makeup_name == "Blush":
                                        processed = applicator.apply_blush(
                                            processed, makeup_mask, makeup_color, makeup_intensity
                                        )
                    
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
                    else:
                        progress_bar.progress(66)
                        status_text.text(f"Step 3/3: Applying {product}...")
                        
                        mask = segmenter.create_mask_for_product(
                            st.session_state.original_image.shape,
                            product,
                            landmarks
                        )
                        
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
                    
<<<<<<< HEAD
                    st.success(f"✅ Makeup applied with BiSeNet! ({st.session_state.processing_time:.2f}s)")
=======
                    if product == "Complete Look":
                        st.success(f"✅ Complete look applied successfully! ({st.session_state.processing_time:.2f}s)")
                    else:
                        st.success(f"✅ {product} applied successfully! ({st.session_state.processing_time:.2f}s)")
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
                    
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
                if product == "Complete Look":
                    caption_text = f"Complete Look: {selected_preset}"
                else:
                    caption_text = f"{product} Applied - {selected_shade} ({intensity}%)"
                
                st.image(
                    cv_to_pil(st.session_state.processed_image),
                    caption=caption_text,
                    use_container_width=True
                )
            
            # ⭐ NEW: Show BiSeNet parsing visualization
            if show_parsing_viz and st.session_state.parsing_map is not None:
                st.subheader("🔬 BiSeNet Face Parsing")
                if hasattr(segmenter, 'bisenet'):
                    parsing_viz = segmenter.bisenet.visualize_parsing(st.session_state.parsing_map)
                    st.image(cv_to_pil(parsing_viz), caption="BiSeNet Segmentation Map", use_container_width=True)
            
            if show_processing_time and st.session_state.processing_time:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(
                        label="Processing Time",
                        value=f"{st.session_state.processing_time:.2f}s",
                        delta="BiSeNet + Deep Learning"
                    )
                with col_m2:
                    segmentation_method = "BiSeNet" if (hasattr(segmenter, 'use_bisenet') and segmenter.use_bisenet) else "Landmarks"
                    st.metric(
                        label="Segmentation Method",
                        value=segmentation_method
                    )
            
            result_pil = cv_to_pil(st.session_state.processed_image)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                if product == "Complete Look":
                    filename = f"trymylook_complete_{selected_preset.lower().replace(' ', '_').replace('&', 'and')}.png"
                else:
                    filename = f"trymylook_{product.lower()}_{selected_shade.lower().replace(' ', '_')}.png"
                
                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name=filename,
                    mime="image/png",
                    use_container_width=True
                )
            
            with col_d2:
                if st.button("🔄 Try Different Settings", use_container_width=True):
                    st.session_state.processed_image = None
                    st.rerun()
            
            if st.session_state.face_data:
                with st.expander("🔍 Detection Details"):
<<<<<<< HEAD
                    details = {
=======
                    details_dict = {
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
                        "Landmarks Detected": len(st.session_state.landmarks),
                        "Face Center": st.session_state.face_data.get('center'),
                        "Face Angle": f"{st.session_state.face_data.get('angle', 0):.2f}°",
                        "Confidence": st.session_state.face_data.get('confidence', 1.0),
<<<<<<< HEAD
                        "Product": product,
                        "Shade": selected_shade,
                        "Intensity": f"{intensity}%",
                        "Segmentation": "BiSeNet" if (hasattr(segmenter, 'use_bisenet') and segmenter.use_bisenet) else "Landmarks"
                    }
                    st.json(details)
=======
                    }
                    
                    if product == "Complete Look":
                        details_dict["Complete Look"] = selected_preset
                        preset_config = get_shades_for_product("Complete Look")[selected_preset]
                        for makeup_type, (shade, int_val) in preset_config.items():
                            details_dict[f"{makeup_type.title()}"] = f"{shade} ({int_val}%)"
                    else:
                        details_dict["Product"] = product
                        details_dict["Shade"] = selected_shade
                        details_dict["Intensity"] = f"{intensity}%"
                    
                    st.json(details_dict)
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
        
        else:
            st.info("👆 Click 'Apply Makeup' to see the result!")
            
<<<<<<< HEAD
            st.markdown("""
            ### 🎨 What happens next?
            1. **Face Detection**: AI finds your face (97% accuracy)
            2. **BiSeNet Parsing**: 19-class semantic segmentation
            3. **Mask Creation**: Pixel-perfect regions for makeup
            4. **Makeup Application**: Realistic blending with texture preservation
            5. **Result**: Professional-quality virtual makeup!
            
            Processing takes 2-4 seconds with BiSeNet on CPU, <1 second on GPU.
            """)
=======
            if product == "Complete Look":
                st.markdown("""
                ### 🎨 Complete Look Feature
                Apply all 4 makeup products at once with professionally coordinated colors:
                - **Foundation** - Even base
                - **Blush** - Natural glow
                - **Eyeshadow** - Eye enhancement
                - **Lipstick** - Statement finish
                
                Choose from 5 pre-designed looks for different occasions!
                """)
            else:
                st.markdown("""
                ### 🎨 What happens next?
                1. **Face Detection**: AI finds your face (97% accuracy)
                2. **Landmark Detection**: 68 precise points mapped
                3. **Segmentation**: Perfect masks for lips/eyes/skin/cheeks
                4. **Makeup Application**: Realistic blending with texture preservation
                5. **Result**: Professional-quality virtual makeup!
                
                Processing takes 2-3 seconds on CPU, <1 second on GPU.
                """)
>>>>>>> 5803f2fa673504d497e546b584ae81ab811fd56d
    
    else:
        st.info("📸 Upload an image first to see results here")
        
        st.markdown("""
        ### 🌟 Features
        - **BiSeNet Face Parsing** (19-class segmentation)
        - **Deep Learning Face Detection** (97% accuracy)
        - **68 Facial Landmarks** for precision
        - **4 Individual Products**: Lipstick, Eyeshadow, Foundation, Blush
        - **Complete Look Feature**: Apply all at once!
        - **25 Professional Shades**
        - **5 Complete Look Presets**
        - **Adjustable Intensity** (0-100%)
        - **Before/After Comparison**
        - **Download Results** in high quality
        
        ### 🚀 Powered By
        - BiSeNet (Bilateral Segmentation Network)
        - Face-Alignment Network (PyTorch)
        - ResNet-50 + Hourglass Architecture
        - Advanced Neural Blending
        """)


st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p style="font-size: 1.1em;"><strong>💄 Trymylook Virtual Makeup Try-On</strong></p>
    <p>Powered by BiSeNet Face Parsing • Deep Learning Pipeline</p>
    <p style="font-size: 0.9em;">Built with Streamlit, PyTorch & OpenCV • Version {}</p>
    <p style="font-size: 0.85em; margin-top: 10px;">
        5 Products • 25 Individual Shades • 5 Complete Looks • For best results: Use well-lit, front-facing photos
    </p>
</div>
""".format(VERSION), unsafe_allow_html=True)


with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Products", len(PRODUCTS))
    if product == "Complete Look":
        st.metric("Preset Selected", selected_preset)
    else:
        st.metric("Current Shade", len(shade_names))
    if st.session_state.processing_time:
        st.metric("Last Processing", f"{st.session_state.processing_time:.2f}s")