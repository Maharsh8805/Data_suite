import streamlit as st
import sys


# Suppress static_ffmpeg to prevent lock file errors
sys.modules["static_ffmpeg"] = None

# Set page configuration
st.set_page_config(
    page_title="Data Cleaner Suite",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism Styling
st.markdown("""
    <style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
    @import url('https://fonts.cdnfonts.com/css/cambria');

    /* Global Typography */
    html, body, [class*="css"], p, h1, h2, h3, h4, h5, h6, span, div, label, button, input, textarea, select {
        font-family: 'Inter', 'Cambria', sans-serif;
        color: #1f2937;
    }
    
    h1, h2, h3 {
        font-family: 'Cambria', 'Playfair Display', serif;
        font-weight: 800;
        letter-spacing: -0.01em;
    }

    /* Modern Background */
    .stApp {
        background: radial-gradient(circle at top left, #f8fafc, #eff6ff, #e0f2fe);
        background-attachment: fixed;
    }

    /* Glassmorphism Container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 3rem 2rem;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.05), 
            0 10px 15px -3px rgba(0, 0, 0, 0.05),
            inset 0 0 0 1px rgba(255,255,255,0.4);
        max-width: 1200px;
        margin-top: 1rem;
    }

    /* Sidebar Styling - HIDDEN */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Premium Header */
    .welcome-header {
        text-align: center;
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.75rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        padding-bottom: 10px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.05));
    }

    .subheader {
        text-align: center;
        color: #64748b;
        font-size: 1.25rem;
        font-weight: 400;
        margin-bottom: 3rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }

    /* Section Headers */
    .section-header {
        font-size: 2rem;
        color: #0f172a;
        text-align: center;
        position: relative;
        padding-bottom: 1rem;
        margin-top: 3rem;
        margin-bottom: 4rem;
    }
    
    /* Header underline removed */

    /* Glass Tool Cards */
    .tool-grid {
        display: flex;
        gap: 24px;
        margin-bottom: 24px;
        justify-content: center;
    }

    .tool-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 24px;
        padding: 2.5rem 1.5rem;
        height: 100%;
        min-height: 460px;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.02),
            0 10px 15px rgba(0, 0, 0, 0.03);
        margin: 0 auto; /* Center card in its column */
        position: relative;
        z-index: 1;
    }
    
    .tool-card::before, .tool-card::after {
        content: none !important;
        display: none !important;
        border: none !important;
    }

    .tool-card:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 
            0 20px 25px -5px rgba(0, 0, 0, 0.1),
            0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: rgba(255, 255, 255, 1);
        outline: none;
        border-top: 1px solid rgba(255,255,255,0.8);
    }
    
    .tool-card h3 {
        font-size: 1.5rem;
        margin: 1rem 0;
        color: #1e293b;
        font-weight: 700;
    }
    
    .tool-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
    }
    
    .tool-card p {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
        text-align: center;
        flex-grow: 1;
        margin-bottom: 1.5rem;
    }

    /* Feature Tags */
    .features {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 1.5rem;
    }

    .feature-tag {
        background: rgba(241, 245, 249, 0.8);
        color: #475569;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }

    /* Call to Action Button inside Card */
    .card-link {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        text-decoration: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        display: inline-block;
        margin-top: auto;
    }

    .card-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(37, 99, 235, 0.3);
    }

    /* Custom Buttons (Global) */
    .stButton>button {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: #1f2937 !important;
        border: 1px solid rgba(255, 255, 255, 1) !important;
        border-radius: 12px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    }

    .stButton>button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #cbd5e1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Primary buttons (Launch Tool etc) need to keep their blue */
    /* We don't use standard st.button for primary actions often, mostly custom HTML cards. 
       But the 'Process' action inside tools is usually Primary. */
    button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
    }

    /* Footer */
    .footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
    }

    /* Utilities */
    .step-number {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .instruction-step {
        display: flex;
        align-items: center;
        background: rgba(255,255,255,0.5);
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.6);
        min-height: 80px;
    }
    
    .instruction-text {
        font-size: 1.05rem;
        color: #334155;
        font-weight: 500;
    }
    
    /* Hide default Streamlit menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state for app mode
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "home"

# Set app mode based on navigation
app_mode = st.session_state.app_mode

# Allow navigation via query parameter links
if "tool" in st.query_params:
    tool_val = st.query_params["tool"]
    mapping = {
        'csv': 'csv_tool',
        'text': 'text_tool',
        'image': 'image_tool',
        'audio': 'audio_tool',
        'video': 'video_tool',
        'home': 'home'
    }
    if tool_val in mapping:
        st.session_state.app_mode = mapping[tool_val]
        app_mode = st.session_state.app_mode
        st.query_params.clear()

# ==================== SIDEBAR ====================


# ==================== HOME PAGE ====================
if app_mode == "home" or app_mode == "🏠 Home":
    # Global Greetings Logic
    st.markdown(f'<h1 class="welcome-header">Welcome to Data Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader" style="text-align: center; max-width: 800px; margin: 0 auto 3rem auto;">Professional tools for data preprocessing. Clean, optimize, and standardize your datasets.</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<h2 class="section-header" style="text-align: center;">Select Your Tool</h2>', unsafe_allow_html=True)
    
    # Tools Data
    tools = [
        {"icon": "📊", "name": "CSV Cleaner", "desc": "Clean and process tabular data with outlier detection and null handling.", "features": ["Statistical Analysis", "Null Handling", "Deduplication"], "link": "?tool=csv"},
        {"icon": "📝", "name": "Text Cleaner", "desc": "Sanitize text data by removing HTML, special characters, and PII.", "features": ["HTML Stripping", "PII Removal", "Encoding Fix"], "link": "?tool=text"},
        {"icon": "🖼️", "name": "Image Cleaner", "desc": "Process images with denoising, format conversion, and metadata removal.", "features": ["Auto-Restore", "EXIF Removal", "Smart Resize"], "link": "?tool=image"}
    ]
    
    media_tools = [
        {"icon": "🎵", "name": "Audio Processor", "desc": "Optimize audio files with voice enhancement and format conversion.", "features": ["Voice EQ", "Silence Trimming", "Format Convert"], "link": "?tool=audio"},
        {"icon": "🎬", "name": "Video Studio", "desc": "Enhance and compress video files with frame processing.", "features": ["Transcoding", "GIF Creation", "Compression"], "link": "?tool=video"}
    ]
    
    # Row 1
    cols = st.columns(3)
    for i, tool in enumerate(tools):
        with cols[i]:
            st.markdown(f"""
            <div class="tool-card">
                <div class="tool-icon">{tool['icon']}</div>
                <h3>{tool['name']}</h3>
                <p>{tool['desc']}</p>
                <div class="features">
                    {''.join([f'<span class="feature-tag">{f}</span>' for f in tool['features']])}
                </div>
                <a class="card-link" href="{tool['link']}">Launch Tool</a>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
            
    # Row 2 (Centered) - Using spacing columns to force center alignment
    # [spacer, card, card, spacer] -> [2, 4, 4, 2] roughly centers them
    c1, c2, c3, c4 = st.columns([2, 5, 5, 2])
    with c2:
        tool = media_tools[0]
        st.markdown(f"""
        <div class="tool-card">
            <div class="tool-icon">{tool['icon']}</div>
            <h3>{tool['name']}</h3>
            <p>{tool['desc']}</p>
            <div class="features">
                {''.join([f'<span class="feature-tag">{f}</span>' for f in tool['features']])}
            </div>
            <a class="card-link" href="{tool['link']}">Launch Tool</a>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        tool = media_tools[1]
        st.markdown(f"""
        <div class="tool-card">
            <div class="tool-icon">{tool['icon']}</div>
            <h3>{tool['name']}</h3>
            <p>{tool['desc']}</p>
            <div class="features">
                {''.join([f'<span class="feature-tag">{f}</span>' for f in tool['features']])}
            </div>
            <a class="card-link" href="{tool['link']}">Launch Tool</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
         st.markdown('<h2 class="section-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
         steps = [
             "Select the appropriate tool from the dashboard.",
             "Upload your raw data files (CSV, Images, Audio, etc).",
             "Configure cleaning parameters or use Auto-Mode.",
             "Preview results instantly and download cleaned assets."
         ]
         for i, step in enumerate(steps, 1):
             st.markdown(f"""
             <div class="instruction-step">
                <div class="step-number">{i}</div>
                <div class="instruction-text">{step}</div>
             </div>
             """, unsafe_allow_html=True)
             
    with c2:
        st.markdown('<h2 class="section-header">Use Cases</h2>', unsafe_allow_html=True)
        use_cases = [
            ("Machine Learning", "Prepare clean datasets for training models"),
            ("Analytics", "Standardize data for BI dashboards"),
            ("Privacy", "Strip metadata and PII from public releases"),
            ("Compression", "Reduce storage costs by optimizing media")
        ]
        for i, (title, desc) in enumerate(use_cases, 1):
            st.markdown(f"""
            <div style="display: flex; align-items: center; background: rgba(255,255,255,0.5); padding: 12px; border-radius: 12px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.6); min-height: 80px;">
                <div class="step-number">{i}</div>
                <div style="flex: 1;">
                    <strong style="color: #1e293b; font-size: 1.05rem;">{title}</strong><br>
                    <span style="color: #334155; font-size: 1.05rem; font-weight: 500;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


    st.markdown("""
        <div class="footer">
            <div style="margin-bottom: 0.5rem;">
                <strong style="font-size: 1rem;">Data Cleaner Suite</strong>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                © 2025 Data Suite. All rights reserved.
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==================== CSV CLEANER ====================
elif app_mode == "csv_tool":
    c1, c2, c3 = st.columns([4, 4, 4])
    with c2:
        if st.button("🏠 Back to Home", key="home_btn_csv", use_container_width=True):
            st.session_state.app_mode = "home"
            st.rerun()
            
        # Navigation Dropdown
        nav_options = {
            "📊 CSV Cleaner": "csv_tool",
            "📝 Text Cleaner": "text_tool",
            "🖼️ Image Cleaner": "image_tool",
            "🎵 Audio Processor": "audio_tool",
            "🎬 Video Studio": "video_tool"
        }
        # Invert map for display
        nav_display = list(nav_options.keys())
        current_idx = nav_display.index("📊 CSV Cleaner")
        
        selected_nav = st.selectbox("Navigate to:", nav_display, index=current_idx, label_visibility="collapsed", key="nav_csv")
        
        if nav_options[selected_nav] != "csv_tool":
            st.session_state.app_mode = nav_options[selected_nav]
            st.rerun()
    try:
        import importlib
        import csv_cleaner
        importlib.reload(csv_cleaner)
        csv_cleaner.render()
    except Exception as e:
        st.error(f"Error loading CSV Cleaner: {e}")

# ==================== TEXT CLEANER ====================
elif app_mode == "text_tool":
    c1, c2, c3 = st.columns([4, 4, 4])
    with c2:
        if st.button("🏠 Back to Home", key="home_btn_text", use_container_width=True):
            st.session_state.app_mode = "home"
            st.rerun()
            
        nav_options = {
            "📝 Text Cleaner": "text_tool",
            "📊 CSV Cleaner": "csv_tool",
            "🖼️ Image Cleaner": "image_tool",
            "🎵 Audio Processor": "audio_tool",
            "🎬 Video Studio": "video_tool"
        }
        nav_display = list(nav_options.keys())
        selected_nav = st.selectbox("Navigate to:", nav_display, index=0, label_visibility="collapsed", key="nav_text")
        
        if nav_options[selected_nav] != "text_tool":
            st.session_state.app_mode = nav_options[selected_nav]
            st.rerun()
    try:
        import importlib
        import text_cleaner
        importlib.reload(text_cleaner)
        text_cleaner.render()
    except Exception as e:
        st.error(f"Error loading Text Cleaner: {e}")

# ==================== IMAGE CLEANER ===================
elif app_mode == "image_tool":
    c1, c2, c3 = st.columns([4, 4, 4])
    with c2:
        if st.button("🏠 Back to Home", key="home_btn_img", use_container_width=True):
            st.session_state.app_mode = "home"
            st.rerun()
            
        nav_options = {
            "🖼️ Image Cleaner": "image_tool",
            "📊 CSV Cleaner": "csv_tool",
            "📝 Text Cleaner": "text_tool",
            "🎵 Audio Processor": "audio_tool",
            "🎬 Video Studio": "video_tool"
        }
        nav_display = list(nav_options.keys())
        selected_nav = st.selectbox("Navigate to:", nav_display, index=0, label_visibility="collapsed", key="nav_img")
        
        if nav_options[selected_nav] != "image_tool":
            st.session_state.app_mode = nav_options[selected_nav]
            st.rerun()
    try:
        import importlib
        import image_cleaner
        importlib.reload(image_cleaner)
        image_cleaner.render()
    except Exception as e:
        st.error(f"Error loading Image Cleaner: {e}")

# ==================== AUDIO CLEANER ===================
elif app_mode == "audio_tool":
    c1, c2, c3 = st.columns([4, 4, 4])
    with c2:
        if st.button("🏠 Back to Home", key="home_btn_audio", use_container_width=True):
            st.session_state.app_mode = "home"
            st.rerun()
            
        nav_options = {
            "🎵 Audio Processor": "audio_tool",
            "📊 CSV Cleaner": "csv_tool",
            "📝 Text Cleaner": "text_tool",
            "🖼️ Image Cleaner": "image_tool",
            "🎬 Video Studio": "video_tool"
        }
        nav_display = list(nav_options.keys())
        selected_nav = st.selectbox("Navigate to:", nav_display, index=0, label_visibility="collapsed", key="nav_audio")
        
        if nav_options[selected_nav] != "audio_tool":
            st.session_state.app_mode = nav_options[selected_nav]
            st.rerun()
    try:
        import importlib
        import audio_cleaner
        importlib.reload(audio_cleaner)
        audio_cleaner.render()
    except Exception as e:
        st.error(f"Error loading Audio Cleaner: {e}")

# ==================== VIDEO CLEANER ===================
elif app_mode == "video_tool":
    c1, c2, c3 = st.columns([4, 4, 4])
    with c2:
        if st.button("🏠 Back to Home", key="home_btn_video", use_container_width=True):
            st.session_state.app_mode = "home"
            st.rerun()
            
        nav_options = {
            "🎬 Video Studio": "video_tool",
            "📊 CSV Cleaner": "csv_tool",
            "📝 Text Cleaner": "text_tool",
            "🖼️ Image Cleaner": "image_tool",
            "🎵 Audio Processor": "audio_tool"
        }
        nav_display = list(nav_options.keys())
        selected_nav = st.selectbox("Navigate to:", nav_display, index=0, label_visibility="collapsed", key="nav_video")
        
        if nav_options[selected_nav] != "video_tool":
            st.session_state.app_mode = nav_options[selected_nav]
            st.rerun()
    try:
        import importlib
        import video_cleaner
        importlib.reload(video_cleaner)
        video_cleaner.render()
    except Exception as e:
        st.error(f"Error loading Video Cleaner: {e}")