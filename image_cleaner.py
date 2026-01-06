import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import zipfile
from pathlib import Path
import pandas as pd
import altair as alt
from PIL.ExifTags import TAGS

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


    
# Try to import skimage for advanced filters (non-ML)
try:
    from skimage import metrics, restoration  # type: ignore[import]
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse  # type: ignore[import]
    from skimage.filters import unsharp_mask  # type: ignore[import]
    from skimage.exposure import equalize_adapthist  # type: ignore[import]
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    ssim = None

# ML Grade Restoration (Traditional Deconvolution only, no heavy ML)
ML_AVAILABLE = SKIMAGE_AVAILABLE

# Set page configuration (only when running this file directly)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Image Data Cleaner",
        page_icon="🖼️",
        layout="wide"
    )

# Custom CSS for premium styling (Matching main_app.py)
st.markdown("""
    <style>
    /* Import Cambria font */
    @import url('https://fonts.cdnfonts.com/css/cambria');
    
    /* Apply Cambria font globally */
    html, body, [class*="css"], p, h1, h2, h3, h4, h5, h6, span, div, label, button, input, textarea, select {
        font-family: 'Cambria', 'Georgia', serif;
    }
    
    /* Main background gradient */
    /* Content area styling */
    .main .block-container {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #333333;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Button styling */
    /* Button styling handled by main_app.py */
    
    /* Tool Switcher styling */
    div[data-testid="stSelectbox"] > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        min-height: 40px; /* Force minimum height */
    }
    
    /* Force specific height on the inner select element */
    div[data-baseweb="select"] > div {
        min-height: 40px !important;
        height: 40px !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 0, 0, 0.05);
        color: #111;
        border-bottom: 2px solid #333;
    }
    
    /* Success/Info Alerts */
    .stSuccess, .stInfo, .stWarning {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid transparent;
    }
    .stSuccess { background-color: rgba(209, 250, 229, 0.5); border-color: #10b981; color: #065f46; }
    .stInfo { background-color: rgba(219, 234, 254, 0.5); border-color: #3b82f6; color: #1e40af; }
    .stWarning { background-color: rgba(254, 243, 199, 0.5); border-color: #f59e0b; color: #92400e; }
    
    </style>
""", unsafe_allow_html=True)

# ----------------- Image Processing Functions -----------------

def get_exif_data(img):
    """Extract EXIF data as a DataFrame"""
    exif_data = {}
    try:
        if hasattr(img, '_getexif') and img._getexif():
            for tag, value in img._getexif().items():
                decoded = TAGS.get(tag, tag)
                # Skip long binary data
                if isinstance(value, bytes) and len(value) > 100:
                    value = "<Binary Data>"
                exif_data[decoded] = str(value)
    except Exception:
        pass
    
    if exif_data:
        return pd.DataFrame(list(exif_data.items()), columns=['Property', 'Value'])
    return pd.DataFrame(columns=['Property', 'Value'])

def get_color_histogram(img):
    """Generate RGB histogram data for Altair"""
    try:
        # Resize for speed if huge
        if img.width > 500:
            img_small = img.resize((500, int(500 * img.height / img.width)))
        else:
            img_small = img
            
        img_arr = np.array(img_small.convert('RGB'))
        
        # Calculate histograms per channel
        hist_r, _ = np.histogram(img_arr[:,:,0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(img_arr[:,:,1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(img_arr[:,:,2], bins=256, range=(0, 256))
        
        # Create DataFrame
        df_hist = pd.DataFrame({
            'Intensity': np.tile(np.arange(256), 3),
            'Count': np.concatenate([hist_r, hist_g, hist_b]),
            'Channel': ['Red']*256 + ['Green']*256 + ['Blue']*256
        })
        return df_hist
    except Exception:
        return None

def validate_image(img_file):
    """Check if image is readable and valid"""
    try:
        img = Image.open(img_file)
        img.verify()
        img_file.seek(0)
        img = Image.open(img_file)
        
        # Handle orientation from EXIF
        try:
            img = ImageOps.exif_transpose(img)
        except (AttributeError, OSError, Exception):
            pass
            
        return True, img
    except Exception:
        return False, None

def remove_metadata(img):
    """Remove EXIF and other metadata by creating a fresh image"""
    data = list(img.getdata())
    image_without_exif = Image.new(img.mode, img.size)
    image_without_exif.putdata(data)
    return image_without_exif

def convert_image_format(img, target_format='RGB'):
    """Convert image to consistent format"""
    if target_format == 'RGB':
        if img.mode != 'RGB': return img.convert('RGB')
    elif target_format == 'L':  # Grayscale
        if img.mode != 'L': return img.convert('L')
    elif target_format == 'RGBA':
        if img.mode != 'RGBA': return img.convert('RGBA')
    return img

def resize_image(img, max_resolution=(1024, 1024)):
    """Resize image if it exceeds max resolution while maintaining aspect ratio"""
    img.thumbnail(max_resolution, Image.Resampling.LANCZOS)
    return img

def upscale_image(img, scale_factor=1.5):
    """Upscale image by a factor"""
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    return img.resize(new_size, Image.Resampling.LANCZOS)

# --- OpenCV Advanced Functions ---

def detect_blur(img_pil):
    """Detect blur using Laplacian variance"""
    if not CV2_AVAILABLE: 
        return False, 0.0
    
    try:
        # Validate image size
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return False, 0.0
            
        img_cv = np.array(img_pil.convert('RGB'))[:, :, ::-1].copy() # PIL RGB -> CV2 BGR
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Handle edge case where variance might be NaN or inf
        if not np.isfinite(variance):
            return False, 0.0
            
        return variance < 100.0, variance
    except Exception:
        return False, 0.0

def apply_denoising(img_pil, strength=10):
    """Apply Non-Local Means Denoising"""
    if not CV2_AVAILABLE: 
        return img_pil
    
    try:
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
            
        img_cv = np.array(img_pil.convert('RGB'))[:, :, ::-1].copy()
        dst = cv2.fastNlMeansDenoisingColored(img_cv, None, h=strength, hColor=strength, templateWindowSize=7, searchWindowSize=21)
        return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    except Exception:
        return img_pil

def apply_sharpening(img_pil, strength=1.0):
    """Apply Unsharp Masking"""
    if not CV2_AVAILABLE: 
        return img_pil
    
    try:
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
            
        img_cv = np.array(img_pil.convert('RGB'))
        gaussian = cv2.GaussianBlur(img_cv, (0, 0), 2.0)
        unsharp = cv2.addWeighted(img_cv, 1.0 + strength, gaussian, -strength, 0)
        return Image.fromarray(unsharp)
    except Exception:
        return img_pil

def apply_detail_enhancement(img_pil):
    """Apply CV2 Detail Enhancement for realism"""
    if not CV2_AVAILABLE:
        return img_pil
        
    try:
        img_cv = np.array(img_pil.convert('RGB'))[:, :, ::-1].copy()
        # sigma_s=10 (neighborhood), sigma_r=0.15 (edge preservation)
        # These values provide clarity without 'cartoonish' look
        dst = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)
        return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    except Exception:
        return img_pil

def apply_clahe(img_pil, clip_limit=2.0):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if not CV2_AVAILABLE: 
        return img_pil
    
    try:
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
            
        img_cv = np.array(img_pil.convert('RGB'))[:, :, ::-1].copy()
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return Image.fromarray(cv2.cvtColor(limg, cv2.COLOR_LAB2RGB))
    except Exception:
        return img_pil

# --- Evaluation Metrics ---

def calculate_quality_metrics(original_img, processed_img):
    """Calculate SSIM, PSNR, and MSE between images"""
    if not SKIMAGE_AVAILABLE or ssim is None:
        return None, None, None
    
    try:
        # Convert to numpy arrays
        orig_arr = np.array(original_img.convert('RGB'))
        proc_arr = np.array(processed_img.convert('RGB'))
        
        # Resize if dimensions don't match
        if orig_arr.shape != proc_arr.shape:
            # Fallback: use cv2 resize
            if CV2_AVAILABLE:
                proc_arr = cv2.resize(proc_arr, (orig_arr.shape[1], orig_arr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            else:
                return None, None, None
        
        # Calculate metrics
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim_val = ssim(orig_arr, proc_arr, channel_axis=2, data_range=255)
            psnr_val = psnr(orig_arr, proc_arr, data_range=255)
            mse_val = mse(orig_arr, proc_arr)
        
        return ssim_val, psnr_val, mse_val
    except Exception:
        return None, None, None

def gaussian_kernel(size=5, sigma=1.0):
    """Generate a Gaussian kernel for PSF"""
    try:
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / g.sum()
    except Exception:
        return np.ones((size, size)) / (size ** 2)

def apply_wiener_deconvolution(img_pil, psf_size=5, balance=0.1):
    """Apply Wiener deconvolution with Gaussian PSF"""
    if not ML_AVAILABLE:
        return img_pil
    
    try:
        # Validate image
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
        
        # Normalize to 0-1 range
        img_arr = np.array(img_pil.convert('RGB')) / 255.0
        
        # Validate array
        if img_arr.size == 0 or len(img_arr.shape) != 3:
            return img_pil
        
        # Create Gaussian PSF (better than box filter)
        psf = gaussian_kernel(size=psf_size, sigma=1.0)
        
        # Apply Wiener deconvolution per channel
        result = np.zeros_like(img_arr)
        for i in range(3):
            result[:, :, i] = restoration.wiener(img_arr[:, :, i], psf, balance=balance)
        
        # Validate result before clipping
        if not np.all(np.isfinite(result)):
            return img_pil
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        return img_pil

def apply_richardson_lucy(img_pil, num_iter=30, psf_size=5):
    """Apply Richardson-Lucy deconvolution for high-accuracy restoration"""
    if not ML_AVAILABLE:
        return img_pil
    
    try:
        # Validate image
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
        
        # Normalize to 0-1 range
        img_arr = np.array(img_pil.convert('RGB')) / 255.0
        
        # Validate array
        if img_arr.size == 0 or len(img_arr.shape) != 3:
            return img_pil
            
        # Create Gaussian PSF
        psf = gaussian_kernel(size=psf_size, sigma=1.0)
        
        # Apply RL per channel
        result = np.zeros_like(img_arr)
        for i in range(3):
            result[:, :, i] = restoration.richardson_lucy(img_arr[:, :, i], psf, num_iter=num_iter, clip=False)
            
        # Validate result before clipping
        if not np.all(np.isfinite(result)):
            return img_pil
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        return img_pil

def apply_tv_denoise(img_pil, weight=0.1):
    """Apply Total Variation denoising (edge-preserving)"""
    if not SKIMAGE_AVAILABLE:
        return img_pil
    
    try:
        # Validate image
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
        
        img_arr = np.array(img_pil.convert('RGB')) / 255.0
        
        # Validate array
        if img_arr.size == 0 or len(img_arr.shape) != 3:
            return img_pil
        
        # Apply TV denoising
        denoised = restoration.denoise_tv_chambolle(img_arr, weight=weight, channel_axis=2)
        
        # Validate result
        if not np.all(np.isfinite(denoised)):
            return img_pil
        
        result = (denoised * 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        return img_pil

def apply_advanced_sharpen(img_pil, radius=2.0, amount=1.5):
    """Apply advanced unsharp masking"""
    if not SKIMAGE_AVAILABLE:
        return img_pil
    
    try:
        # Validate image
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
        
        img_arr = np.array(img_pil.convert('RGB')) / 255.0
        
        # Validate array
        if img_arr.size == 0 or len(img_arr.shape) != 3:
            return img_pil
        
        # Apply unsharp mask
        sharpened = unsharp_mask(img_arr, radius=radius, amount=amount, channel_axis=2)
        
        # Validate result
        if not np.all(np.isfinite(sharpened)):
            return img_pil
        
        result = (np.clip(sharpened, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        return img_pil

def apply_adaptive_equalization(img_pil, clip_limit=0.03):
    """Apply adaptive histogram equalization (CLAHE alternative)"""
    if not SKIMAGE_AVAILABLE:
        return img_pil
    
    try:
        # Validate image
        if img_pil.size[0] == 0 or img_pil.size[1] == 0:
            return img_pil
        
        img_arr = np.array(img_pil.convert('RGB')) / 255.0
        
        # Validate array
        if img_arr.size == 0 or len(img_arr.shape) != 3:
            return img_pil
        
        # Apply adaptive equalization
        equalized = equalize_adapthist(img_arr, clip_limit=clip_limit)
        
        # Validate result
        if not np.all(np.isfinite(equalized)):
            return img_pil
        
        result = (equalized * 255).astype(np.uint8)
        return Image.fromarray(result)
    except Exception:
        return img_pil

def auto_clean_image(img, skip_denoise=False, skip_sharpen=False):
    """Automatically detect and fix common image issues"""
    steps = []
    
    if not CV2_AVAILABLE:
        return img, steps
    
    # Validate image
    if img.size[0] == 0 or img.size[1] == 0:
        return img, steps
    
    # Detect blur
    is_blurry, _ = detect_blur(img)
    
    # 1. Skip Denoising to preserve texture (Avoid "plastic/waxy" look)
    # img = apply_denoising(img, strength=5)
    
    # 2. Skip Contrast Enhancement to preserve natural colors
    # img = apply_clahe(img, clip_limit=2.0)
    
    # 3. If blurry, apply restoration
    # (User request: "looks like real image not edited")
    # img = apply_clahe(img, clip_limit=2.0)
    
    # 3. If blurry, apply restoration
    if is_blurry:
        # Upscale only if strictly necessary (very small images)
        if img.width < 500:
            img = upscale_image(img, 1.5)
            steps.append("⬆️ Auto-Upscaled (Small image)")
        
        # Gentle sharpen to recover focus
        if not skip_sharpen:
            img = apply_sharpening(img, strength=0.5)
            steps.append("🔪 Sharpened (Gentle)")
    
    # 4. For already clear images, DO NOTHING or very light touch
    # User wants "realistic" -> No edits if not needed.
    elif not skip_sharpen:
        # Very subtle sharpen just to crispen edges slightly
        img = apply_sharpening(img, strength=0.2)
        steps.append("✨ Clarified (Subtle)")
    
    return img, steps

def process_single_image(uploaded_file, options):
    """Process a single image and return result dict"""
    is_valid, img = validate_image(uploaded_file)
    if not is_valid:
        return {'error': 'Invalid image'}
    
    # Keep original for comparison
    original_img = img.copy()
    original_size = img.size
    steps = []
    ssim_val, psnr_val, mse_val = None, None, None
    
    # AUTO-CLEAN MODE: Always apply automatic cleaning first
    if options.get('auto_clean', True):
        # Standard auto-clean
        skip_sharpen = options.get('use_ml', False) # use_ml logic might need cleanup too, setting default false
        img, auto_steps = auto_clean_image(img, skip_denoise=False, skip_sharpen=False)
        steps.extend(auto_steps)
    
    # Traditional Filters (Manual)
    if not options.get('auto_clean', True):
        # Denoising & Smoothing
        if options.get('denoise'):
            img = apply_denoising(img, options['denoise_strength'])
            steps.append(f"Denoised (Str: {options['denoise_strength']})")
        if options.get('median_filter'):
            img = apply_median_filter(img, options['median_size'])
            steps.append(f"Median Filter (Size: {options['median_size']})")
        if options.get('gaussian_blur'):
            img = apply_gaussian_blur(img, options['gaussian_size'], options['gaussian_sigma'])
            steps.append(f"Gaussian Blur (Size: {options['gaussian_size']})")
        if options.get('bilateral_filter'):
            img = apply_bilateral_filter(img, options['bilateral_d'], options['bilateral_color'], options['bilateral_space'])
            steps.append("Bilateral Filter Applied")
            
        # Detail & Contrast
        if options.get('sharpen'):
            img = apply_sharpening(img, options['sharpen_strength'])
            steps.append(f"Sharpened (Str: {options['sharpen_strength']})")
        if options.get('clahe'):
            img = apply_clahe(img, options['clahe_strength'])
            steps.append(f"CLAHE Contrast (Limit: {options['clahe_strength']})")
        if options.get('edge_detect'):
            img = apply_edge_detection(img, options['edge_method'])
            steps.append(f"Edge Detection ({options['edge_method']})")
        if options.get('normalize'):
            img = normalize_image(img, options['norm_method'])
            steps.append(f"Normalized ({options['norm_method']})")

    # Format & Size
    if options.get('convert_format', True):
        img = convert_image_format(img, options.get('target_format', 'RGB'))
        steps.append(f"Converted to {options.get('target_format', 'RGB')}")
        
    if options.get('resize', True):
        img = resize_image(img, options.get('max_resolution', (1024, 1024)))
        if img.size != original_size:
            steps.append(f"Resized to {img.size}")
            
    if options.get('remove_metadata', True):
        img = remove_metadata(img)
        steps.append("Metadata Removed")
        
    # Quality Metrics
    if SKIMAGE_AVAILABLE:
        ssim_val, psnr_val, mse_val = calculate_quality_metrics(original_img, img)

    # Output Preparation
    img_byte_arr = io.BytesIO()
    save_fmt = options.get('output_format', 'JPEG').upper()
    if save_fmt == 'JPG': save_fmt = 'JPEG'
    
    try:
        img.save(img_byte_arr, format=save_fmt, quality=95)
        img_byte_arr.seek(0)
    except Exception as e:
        return {'error': f'Failed to save image: {str(e)}'}
    
    return {
        'filename': uploaded_file.name,
        'original_img': original_img,
        'processed_img': img,
        'data': img_byte_arr.getvalue(),
        'steps': steps,
        'original_size': original_size,
        'processed_size': img.size,
        'ssim': ssim_val,
        'psnr': psnr_val,
        'mse': mse_val,
        'exif': get_exif_data(original_img),
        'histogram_data': get_color_histogram(img)
    }

def render():
    # Navigation is handled by the Global Sidebar in main_app.py
    st.markdown("---")
    st.title("🖼️ Image Processor Pro")
    st.write("Top-notch image restoration and standardization tool.")

    uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"📂 Loaded {len(uploaded_files)} images")
        
        # Settings Layout
        st.subheader("⚙️ Processing Settings")
        

        tab_restoration, tab_format, tab_output = st.tabs(["✨ Restoration", "📏 Format & Privacy", "💾 Output"])
        
        with tab_restoration:
            auto_clean = st.checkbox("🪄 Automatic Cleaning (Recommended)", value=True, help="Automatically detects and fixes blur, noise, and contrast issues")
            
            if auto_clean:
                st.info("✨ **Automatic Mode**: Basic cleaning (denoise, sharpen, contrast) will be applied.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### **Denoising & Smoothing**")
                denoise = st.checkbox("🧹 Fast Denoise", value=False, disabled=auto_clean)
                denoise_str = st.slider("Denoise Strength", 1, 20, 5, disabled=not denoise or auto_clean)
                
                median_filter = st.checkbox("🔲 Median Filter", value=False, disabled=auto_clean)
                median_size = st.selectbox("Kernel Size (Median)", [3, 5, 7], index=1, disabled=not median_filter or auto_clean)
                
                gaussian_blur = st.checkbox("🌫️ Gaussian Blur", value=False, disabled=auto_clean)
                gaussian_size = st.selectbox("Kernel Size (Gaussian)", [3, 5, 7], index=1, disabled=not gaussian_blur or auto_clean)
                gaussian_sigma = st.slider("Sigma (Blur)", 0.0, 5.0, 1.0, disabled=not gaussian_blur or auto_clean)
                
                bilateral = st.checkbox("💎 Bilateral Filter", value=False, disabled=auto_clean)
                bilateral_d = st.slider("Diameter (Bilateral)", 1, 20, 9, disabled=not bilateral or auto_clean)
            
            with col2:
                st.markdown("#### **Enhancement & Analysis**")
                sharpen = st.checkbox("🔪 Sharpen", value=False, disabled=auto_clean)
                sharpen_str = st.slider("Sharpen Amount", 0.1, 3.0, 0.8, disabled=not sharpen or auto_clean)
                
                clahe = st.checkbox("🌗 Adaptive Contrast (CLAHE)", value=False, disabled=auto_clean)
                clahe_str = st.slider("Clip Limit", 1.0, 5.0, 2.0, disabled=not clahe or auto_clean)
                
                edge_detect = st.checkbox("📐 Edge Detection", value=False, disabled=auto_clean)
                edge_method = st.selectbox("Method", ["canny", "sobel"], disabled=not edge_detect or auto_clean)
                
                normalize = st.checkbox("⚖️ Normalize", value=False, disabled=auto_clean)
                norm_method = st.selectbox("Normalization Type", ["min-max", "z-score"], disabled=not normalize or auto_clean)

        with tab_format:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### **Dimensions**")
                resize = st.checkbox("Resize Images", value=True)
                max_dim = st.number_input("Max Dimension (px)", 128, 8192, 1024, disabled=not resize)
                
            with col2:
                st.markdown("#### **Format & Privacy**")
                convert_fmt = st.checkbox("Convert Color Mode", value=True)
                target_fmt = st.selectbox("Mode", ['RGB', 'L (Grayscale)', 'RGBA'], disabled=not convert_fmt).split()[0]
                remove_meta = st.checkbox("🛡️ Remove Metadata (EXIF)", value=True, help="Strips GPS and camera data for privacy.")

        with tab_output:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### **File Format**")
                out_fmt = st.selectbox("Save As", ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF'])
                
            with col2:
                st.markdown("#### **Naming**")
                rename = st.checkbox("Rename Files", value=True)
                prefix = st.text_input("Prefix", "img", disabled=not rename)

        if st.button("🚀 Process Images", type="primary", width="stretch"):
            options = {
                'auto_clean': auto_clean,
                'denoise': denoise,
                'denoise_strength': denoise_str,
                'median_filter': median_filter,
                'median_size': median_size,
                'gaussian_blur': gaussian_blur,
                'gaussian_size': gaussian_size,
                'gaussian_sigma': gaussian_sigma,
                'bilateral_filter': bilateral,
                'bilateral_d': bilateral_d,
                'bilateral_color': 75,
                'bilateral_space': 75,
                'sharpen': sharpen,
                'sharpen_strength': sharpen_str,
                'clahe': clahe,
                'clahe_strength': clahe_str,
                'edge_detect': edge_detect,
                'edge_method': edge_method,
                'normalize': normalize,
                'norm_method': norm_method,
                'resize': resize,
                'max_resolution': (max_dim, max_dim),
                'convert_format': convert_fmt,
                'target_format': target_fmt,
                'remove_metadata': remove_meta,
                'output_format': out_fmt,
                'rename': rename,
                'prefix': prefix
            }
            
            with st.spinner("Processing..."):
                results = []
                for idx, f in enumerate(uploaded_files):
                    res = process_single_image(f, options)
                    if 'error' not in res:
                        if rename:
                            ext = out_fmt.lower()
                            res['filename'] = f"{prefix}_{idx+1:03d}.{ext}"
                        results.append(res)
                
                st.session_state['img_results'] = results
                st.rerun()

    # Results View
    if 'img_results' in st.session_state:
        results = st.session_state['img_results']
        st.markdown("---")
        
        # Header + Reset
        row1_1, row1_2 = st.columns([4, 1])
        with row1_1: st.subheader(f"📊 Results ({len(results)} images)")
        with row1_2: 
            if st.button("🔄 Start Over", type="secondary"):
                del st.session_state['img_results']
                st.rerun()

        # Quality Metrics Summary (if ML mode was used)
        if any(r.get('ssim') is not None for r in results):
            st.markdown("### 📈 Quality Metrics Summary")
            valid_results = [r for r in results if r.get('ssim') is not None]
            avg_ssim = np.mean([r['ssim'] for r in valid_results])
            avg_psnr = np.mean([r['psnr'] for r in valid_results])
            avg_mse = np.mean([r['mse'] for r in valid_results])
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Avg SSIM", f"{avg_ssim:.4f}", 
                         delta="Excellent" if avg_ssim > 0.9 else "Good" if avg_ssim > 0.8 else "Fair")
            with m2:
                st.metric("Avg PSNR", f"{avg_psnr:.2f} dB",
                         delta="High Quality" if avg_psnr > 30 else "Good" if avg_psnr > 25 else "Fair")
            with m3:
                st.metric("Avg MSE", f"{avg_mse:.2f}",
                         delta="Low Error" if avg_mse < 100 else "Moderate")
            with m4:
                quality_score = "🌟 Excellent" if avg_ssim > 0.9 and avg_psnr > 30 else "✅ Good" if avg_ssim > 0.8 else "⚠️ Fair"
                st.metric("Overall Quality", quality_score)
            
            st.caption("**SSIM**: Structural Similarity (higher is better, max 1.0) | **PSNR**: Peak Signal-to-Noise Ratio (higher is better) | **MSE**: Mean Squared Error (lower is better)")
            st.markdown("---")


        st.subheader("👁️ Visual Inspection & Comparison")
        
        # Selector for comparison
        img_names = [r['filename'] for r in results]
        selected_idx = st.selectbox("Select image to inspect:", range(len(results)), format_func=lambda x: img_names[x])
        
        sel_res = results[selected_idx]

        # Renaming Feature
        new_name = st.text_input("Filename", value=sel_res['filename'], key=f"rename_{selected_idx}")
        if new_name != sel_res['filename']:
            sel_res['filename'] = new_name
            # Force update in list to ensure download zip gets it
            results[selected_idx]['filename'] = new_name
            st.rerun()
        
        # Visual Diff UI
        c1, c2 = st.columns(2)
        with c1:
            st.image(sel_res['original_img'], caption=f"Original ({sel_res['original_size'][0]}x{sel_res['original_size'][1]})", width='stretch')
        with c2:
            st.image(sel_res['processed_img'], caption=f"Processed ({sel_res['processed_size'][0]}x{sel_res['processed_size'][1]})", width='stretch')
            
        with st.expander("🛠️ Applied Operations", expanded=True):
            for step in sel_res['steps']:
                st.write(f"- {step}")

        # Deep Analysis Tab
        st.subheader("🔬 Deep Analysis")
        an_tab1, an_tab2 = st.tabs(["📊 Color Distribution (RGB)", "ℹ️ EXIF Metadata"])
        
        with an_tab1:
            if sel_res['histogram_data'] is not None:
                hist_chart = alt.Chart(sel_res['histogram_data']).mark_area(
                    opacity=0.5,
                    interpolate='monotone'
                ).encode(
                    x=alt.X('Intensity', title='Pixel Intensity (0-255)'),
                    y=alt.Y('Count', title=None, stack=None),
                    color=alt.Color('Channel', scale=alt.Scale(domain=['Red', 'Green', 'Blue'], range=['#ef4444', '#22c55e', '#3b82f6'])),
                    tooltip=['Channel', 'Intensity', 'Count']
                ).properties(
                    height=250,
                    width='container',
                    title="RGB Color Histogram"
                ).interactive()
                st.altair_chart(hist_chart, use_container_width=True)
                st.caption("Visualizes the distribution of colors. Well-balanced images usually spread across the full range.")
            else:
                st.info("Histogram not available.")
                
        with an_tab2:
            if not sel_res['exif'].empty:
                st.dataframe(sel_res['exif'], width=1000, height=300, hide_index=True)
            else:
                st.info("No EXIF metadata found in the original image.")

        st.markdown("---")
        # Download All
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            seen_names = set()
            for r in results:
                name = r['filename']
                if name in seen_names:
                    base, ext = os.path.splitext(name)
                    counter = 1
                    while f"{base}_{counter}{ext}" in seen_names:
                        counter += 1
                    name = f"{base}_{counter}{ext}"
                seen_names.add(name)
                zf.writestr(name, r['data'])
        st.download_button("📦 Download All Processed Images (ZIP)", zip_buffer.getvalue(), "processed_images.zip", "application/zip", type="primary", width="stretch")
    
    else:
        st.info("👆 Please upload image files to get started")
        
        st.markdown("""
        ### 📋 How to Use Image Cleaner Pro
        
        #### **Step 1: Upload Images**
        - Click the file uploader above
        - Supports: JPG, JPEG, PNG, BMP, WEBP, TIFF
        - Multiple files can be uploaded at once
        
        #### **Step 2: Choose Processing Options**
        Configure your image processing pipeline:
        

        
        **✨ Restoration & Enhancement:**
        - **Auto-Clean**: Intelligent one-click restoration for blur and noise.
        - **Advanced Filters**: Median (dead pixels), Gaussian (smoothness), and Bilateral (edge-preserving).
        - **Edge Detection**: Highlight defects using Canny or Sobel methods.
        - **Normalization**: Standardize pixel intensities (Min-Max or Z-Score).
        
        **🎨 Format & Privacy:**
        - **Resize Images**: Scale images to max dimensions while preserving aspect ratio.
        - **Strip Metadata**: Remove EXIF data (GPS, camera info) for privacy.
        - **Format Conversion**: Convert to JPEG, PNG, WEBP, BMP, or TIFF.
        - **Color Mode**: Full Color (RGB), Grayscale (L), or Alpha (RGBA).
        
        **📦 Workflow Control:**
        - **Smart Naming**: Rename files with custom prefixes.
        - **Quality Metrics**: SSIM, PSNR, and MSE calculated for every process.
        
        #### **Step 3: Process & Compare**
        - Click "🚀 Process Images"
        - View side-by-side comparison (Original vs Processed)
        - Review applied operations and quality scores.
        
        #### **Step 4: Download**
        - Download individual images or all as a ZIP archive.
        
        ### 💡 Pro Tips
        - Use **Bilateral Filtering** to smooth skin or surfaces while keeping sharp edges.

        - **WEBP format** is highly recommended for web optimization.
        """)

if __name__ == "__main__":
    render()
