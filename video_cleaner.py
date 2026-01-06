# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import io
import zipfile
from pathlib import Path
import tempfile
import os
import json
import subprocess
import time
from PIL import Image
import pandas as pd
import altair as alt

# Try imports
try:
    import cv2
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

# Import ffmpeg manager if available
try:
    import ffmpeg_manager
except ImportError:
    ffmpeg_manager = None

# Set page configuration
if __name__ == "__main__":
    st.set_page_config(
        page_title="Video Data Cleaner",
        page_icon="🎬",
        layout="wide"
    )

# --- Premium CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/cambria');
    
    html, body, [class*="css"], p, h1, h2, h3, h4, h5, h6, span, div, label, button, input, textarea, select {
        font-family: 'Cambria', 'Georgia', serif;
    }
    
    /* Content area styling */
    .main .block-container {
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    
    h1, h2, h3 {
        color: #333333;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Button styling handled by main_app.py */

    .result-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

if VIDEO_SUPPORT:
    # --- Helper Functions ---
    def get_ffmpeg():
        if ffmpeg_manager:
            path, _ = ffmpeg_manager.get_ffmpeg_paths()
            return path
        return "ffmpeg" # Fallback to system path

    def get_video_metadata(video_path):
        cap = cv2.VideoCapture(video_path)
        meta = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0
        }
        if meta['fps'] > 0:
            meta['duration'] = meta['count'] / meta['fps']
        cap.release()
        return meta

    def enhance_frame(frame, method='standard', color_correction=False):
        """Apply Enhancement Filters (Standard vs Pro)"""
        enhanced = frame.copy()
        
        # 1. Denoising
        if method == 'pro':
             # Non-Local Means Denoising (Slow but Studio Quality)
             # h=10 (luminance), hColor=10 (color)
             enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        else:
             # Standard Bilateral (Fast)
             enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
             
        # 2. Color Correction (Auto Levels via CLAHE on L-channel)
        if color_correction:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Clip limit 3.0 gives more 'pop' than 2.0
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            
        # 3. Smart Sharpening (Unsharp Mask)
        # Only apply if not pro-denoising (pro denoise already preserves edges well, over-sharpening looks fake)
        # But a little bit helps. 
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced

    def get_color_histogram(frame):
        """Generate RGB Histogram for a single frame"""
        # Split channels
        chans = cv2.split(frame)
        colors = ('b', 'g', 'r')
        data = []
        
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            # Normalize
            cv2.normalize(hist, hist)
            
            for i, val in enumerate(hist):
                if color == 'r': c_name = 'Red'
                elif color == 'g': c_name = 'Green'
                else: c_name = 'Blue'
                
                data.append({
                    'Intensity': i,
                    'Count': float(val),
                    'Channel': c_name
                })
        return pd.DataFrame(data)

    def process_video_cv2(input_path, output_path, options, preview=False):
        """Standard processing using OpenCV (Best for frame manipulation)"""
        cap = cv2.VideoCapture(input_path)
        
        # Target Props
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        
        tgt_w = orig_w
        tgt_h = orig_h
        
        # Smart Resizing Logic
        preset = options.get('resize_preset')
        if preset:
            # Defined presets (Landscape base)
            presets = {
                "720p HD": (1280, 720),
                "1080p Full HD": (1920, 1080),
                "480p SD": (854, 480)
            }
            if preset in presets:
                p_w, p_h = presets[preset]
                # Check orientation
                if orig_h > orig_w:
                    # Portrait: Swap dimensions
                    tgt_w, tgt_h = p_h, p_w
                else:
                    # Landscape or Square
                    tgt_w, tgt_h = p_w, p_h
        else:
            # Fallback for direct width/height if ever used
            if options.get('width'): tgt_w = options.get('width')
            if options.get('height'): tgt_h = options.get('height')
        
        tgt_fps = options.get('fps')
        if tgt_fps is None: tgt_fps = orig_fps
        
        # Safety fallback for missing FPS in metadata
        if tgt_fps <= 0: tgt_fps = 30.0
        
        # Limit for preview
        max_duration = 5.0 if preview else float('inf')
        max_frames = int(max_duration * tgt_fps) if preview else float('inf')
        
        # Writer
        # Try finding a working codec
        # Try H.264 (avc1) first - Best for Browser
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, tgt_fps, (tgt_w, tgt_h))
        
        if not out.isOpened():
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, tgt_fps, (tgt_w, tgt_h))
            
        if not out.isOpened():
             return 0 # Fail signal

        
        processed = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Resize
                if tgt_w != orig_w or tgt_h != orig_h:
                    frame = cv2.resize(frame, (tgt_w, tgt_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Enhance
                if options.get('enhance'):
                    method = 'pro' if options.get('pro_denoise') else 'standard'
                    frame = enhance_frame(frame, method=method, color_correction=options.get('color_correct'))
                
                out.write(frame)
                processed += 1
                
                if processed >= max_frames:
                    break
        finally:
            cap.release()
            out.release()
            
        return processed

    def process_ffmpeg(input_path, output_path, options, preview=False):
        """Advanced processing using FFmpeg (Best for Audio/GIF/Compression)"""
        ffmpeg_bin = get_ffmpeg()
        if not ffmpeg_bin: return False, "FFmpeg not found"
        
        cmd = [ffmpeg_bin, '-y', '-i', input_path]
        
        # Preview trim
        if preview:
            cmd.extend(['-t', '5'])
            
        # Operations
        if options.get('type') == 'gif':
            # High quality GIF palette gen
            # We assume output_path ends in .gif
            # Complex filter for palette
            filter_str = f"fps={options.get('fps', 15)},scale={options.get('width', 480)}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
            cmd.extend(['-vf', filter_str])
            cmd.append(output_path)
            
        elif options.get('type') == 'remove_audio':
            cmd.extend(['-c:v', 'copy', '-an', output_path])
            
        elif options.get('type') == 'compress':
            # CRF 28 for compression
            cmd.extend(['-vcodec', 'libx264', '-crf', '28', output_path])
            
        else:
            return False, "Unknown FFmpeg op"
            
        # Run
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, str(e)

    def process_single_video(uploaded_file, options):
        # Temp Input
        ext = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
            tmp_in.write(uploaded_file.read())
            in_path = tmp_in.name
            
        # Temp Output
        out_ext = '.gif' if options.get('gif') else '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=out_ext) as tmp_out:
            out_path = tmp_out.name
            
        try:
            # Metadata
            orig_meta = get_video_metadata(in_path)
            
            # --- Capture Analysis Frame (Middle of video) ---
            analysis_frame_orig = None
            analysis_frame_proc = None
            
            cap = cv2.VideoCapture(in_path)
            mid_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                analysis_frame_orig = frame
            cap.release()
            
            # Logic Router
            is_preview = options.get('preview', False)
            
            # 1. SPECIAL: GIF Conversion
            if options.get('gif'):
                # FFmpeg for GIF
                opts = {'type': 'gif', 'fps': 15, 'width': 480} # Standard GIF settings
                ok, err = process_ffmpeg(in_path, out_path, opts, preview=is_preview)
                if not ok: raise Exception(err)
                
            # 2. SPECIAL: Audio Removal
            elif options.get('remove_audio'):
                # FFmpeg for stripping
                opts = {'type': 'remove_audio'}
                ok, err = process_ffmpeg(in_path, out_path, opts, preview=is_preview)
                if not ok: raise Exception(err)
            
            # 3. SPECIAL: Compression (Simple)
            elif options.get('compress'):
                 opts = {'type': 'compress'}
                 ok, err = process_ffmpeg(in_path, out_path, opts, preview=is_preview)
                 if not ok: raise Exception(err)

            # 4. STANDARD: Resize / Enhance / FPS (OpenCV)
            else:
                # Use CV2 for pixel-level control
                # Step 1: Process Video ONLY (OpenCV drops audio)
                # We write to a temp video-only file first
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid_only:
                    vid_only_path = tmp_vid_only.name
                
                processed_count = process_video_cv2(in_path, vid_only_path, options, preview=is_preview)
                
                # Step 2: Merge Audio back (if available)
                ffmpeg_bin = get_ffmpeg()
                merged = False
                
                if ffmpeg_bin and processed_count > 0:
                    try:
                        # Attempt to merge audio from original (in_path) to processed video (vid_only_path)
                        # -map 0:v:0 -> Take video from first input (processed)
                        # -map 1:a:0 -> Take audio from second input (original)
                        # -c copy -> Stream copy (lossless, instant)
                        # -shortest -> Stop when video ends (preview handles 5s)
                        cmd = [ffmpeg_bin, '-y', '-i', vid_only_path, '-i', in_path, 
                               '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', 
                               '-shortest', out_path]
                        
                        # Note: we re-encode audio to aac just in case, or we can use copy. 
                        # Using 'aac' is safer for compatibility if container changed.
                        # But 'copy' is best for original quality. Let's try 'copy' first, fallback to aac?
                        # Let's stick to safe AAC encoding for web compatibility.
                        
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                            merged = True
                    except Exception:
                        # If merge fails (maybe no audio stream?), we fall back to video only
                        pass
                
                if not merged:
                    # Just move the video-only file to out_path
                    import shutil
                    shutil.move(vid_only_path, out_path)
                else:
                    # Clean up temp video
                    if os.path.exists(vid_only_path): os.unlink(vid_only_path)
            
            # Read Result
            with open(out_path, 'rb') as f:
                data = f.read()
                
            new_meta = get_video_metadata(out_path)
            
            # --- Capture Processed Analysis Frame ---
            if analysis_frame_orig is not None:
                # We try to get the approx same frame from output
                cap = cv2.VideoCapture(out_path)
                # Recalc mid based on new count (incase FPS changed, frame count matches duration usually)
                mid_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if ret:
                    analysis_frame_proc = frame
                cap.release()
            
            # --- Prepare Original for Display ---
            # KEY FIX: Browsers (Chrome/Edge) often FAIL to play MP4s if they are not H.264 (e.g. MPEG-4 Visual).
            # We MUST ensure the "Original" shown in the UI is H.264.
            # We now ALWAYS attempt to transcode/remux a display copy, regardless of input extension.
            
            display_original_path = in_path
            temp_display_path = in_path + "_display.mp4"
            
            ffmpeg_bin = get_ffmpeg()
            conversion_successful = False
            
            if ffmpeg_bin:
                # Command to force H.264 / AAC which provides max browser compatibility
                # -preset ultrafast for speed
                # -y to overwrite
                cmd = [ffmpeg_bin, '-y', '-i', in_path, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', temp_display_path]
                
                if is_preview: 
                     # Match the preview duration of the processed video (5s) so side-by-side makes sense
                     cmd.insert(4, '-t')
                     cmd.insert(5, '5')
                     
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if os.path.exists(temp_display_path) and os.path.getsize(temp_display_path) > 0:
                        display_original_path = temp_display_path
                        conversion_successful = True
                except Exception as e:
                    # If ffmpeg fails, we just show the original and hope for the best
                    pass 
            
            # Read the (potentially converted) original data for the UI
            with open(display_original_path, 'rb') as f:
                original_data = f.read()
                
            # Cleanup temp display file
            if conversion_successful and os.path.exists(temp_display_path):
                os.unlink(temp_display_path)
            
            return {
                'filename': uploaded_file.name,
                'data': data,
                'original_data': original_data,
                'orig_meta': orig_meta,
                'new_meta': new_meta,
                'ext': out_ext,
                'preview': is_preview,
                'hist_orig': get_color_histogram(analysis_frame_orig) if analysis_frame_orig is not None else None,
                'hist_proc': get_color_histogram(analysis_frame_proc) if analysis_frame_proc is not None else None
            }
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            if os.path.exists(in_path): os.unlink(in_path)
            if os.path.exists(out_path): os.unlink(out_path)


def render():
    # Navigation is handled by the Global Sidebar in main_app.py
            
    st.markdown("---")
    st.title("🎬 Video Studio Pro")
    st.write("Professional video enhancement, compression, and conversion.")
    
    if not VIDEO_SUPPORT:
        st.error("⚠️ OpenCV missing. Install: pip install opencv-python")
        return

    # Upload
    uploaded_files = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi', 'mkv'], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"📂 {len(uploaded_files)} files loaded")
        
        # Settings
        st.subheader("🎛️ Processing Settings")
        
        # Tabs for different "modes" to keep UI clean
        mode = st.radio("Select Mode:", ["Standard Processing", "Creative Tools (GIF/Audio)", "Compression"], horizontal=True)
        
        final_opts = {}
        
        if mode == "Standard Processing":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### **Resolution & FPS**")
                resize = st.checkbox("Resize Video", value=False)
                res_preset = st.selectbox("Preset", ["720p HD", "1080p Full HD", "480p SD"], disabled=not resize)
                
                chg_fps = st.checkbox("Change FPS", value=False)
                tgt_fps = st.selectbox("Target FPS", [24, 30, 60], index=1, disabled=not chg_fps)
                
            with c2:
                st.markdown("#### **Quality Enhancement**")
                enhance = st.checkbox("🔮 Auto-Enhance (Denoise & Sharpen)", value=True, help="Apply Bilateral Cleaning (Denoising) and Smart Sharpening to fix blurry/grainy videos.")
                
                pro_denoise = st.checkbox("🧪 Pro Denoise (Slow)", value=False, help="Uses Non-Local Means Denoising. Much cleaner but 5x slower.")
                color_corr = st.checkbox("🎨 Auto-Color Correction", value=True, help="Equalize colors for vibrant look.")
                
            final_opts.update({
                'resize_preset': res_preset if resize else None, 
                'fps': tgt_fps if chg_fps else None, 
                'enhance': enhance,
                'pro_denoise': pro_denoise if enhance else False,
                'color_correct': color_corr if enhance else False
            })
            
        elif mode == "Creative Tools (GIF/Audio)":
            c1, c2 = st.columns(2)
            with c1:
                is_gif = st.checkbox("🎞️ Convert to GIF", help="Create animated GIF (15fps, 480p)")
            with c2:
                no_audio = st.checkbox("🔇 Remove Audio", help="Strip audio track (Video Only)")
                
            final_opts.update({'gif': is_gif, 'remove_audio': no_audio})
            
        elif mode == "Compression":
            st.info("Compress video file size while maintaining good quality (CRF 28).")
            do_compress = st.checkbox("Enable Compression", value=True)
            final_opts.update({'compress': do_compress})

        # Global Options
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            preview = st.checkbox("👀 Preview Mode (First 5s)", value=True, help="Process only start of video to test settings.")
            
        final_opts.update({'preview': preview})

        # Action
        if st.button("🚀 Process Videos", type="primary", use_container_width=True):
            with st.spinner("Processing videos..."):
                results = []
                errors = []
                for idx, f in enumerate(uploaded_files):
                    res = process_single_video(f, final_opts)
                    if 'error' not in res:
                        # Default name handled by function or original name could be used
                        # We'll just use the uploaded filename but with new extension
                        base = os.path.splitext(f.name)[0]
                        res['filename'] = f"{base}_processed{res['ext']}"
                        results.append(res)
                    else:
                        errors.append(f"Error processing {f.name}: {res['error']}")
                
                # Update session state
                if results:
                    st.session_state['video_results'] = results
                
                if errors:
                    st.session_state['processing_errors'] = errors
                elif 'processing_errors' in st.session_state:
                    del st.session_state['processing_errors']
                    
                st.rerun()

    # Results
    if 'processing_errors' in st.session_state:
        for err in st.session_state['processing_errors']:
            st.error(err)

    if 'video_results' in st.session_state:
        results = st.session_state['video_results']
        st.markdown("---")
        
        # Header + Reset
        r1, r2 = st.columns([4, 1])
        with r1: st.subheader(f"📊 Results ({len(results)} files)")
        with r2: 
            if st.button("🔄 Start Over", type="secondary"):
                if 'video_results' in st.session_state: del st.session_state['video_results']
                if 'processing_errors' in st.session_state: del st.session_state['processing_errors']
                st.rerun()

        # Visual Comparison Section
        st.subheader("🎬 Video Comparison")
        
        # Selector
        file_names = [r['filename'] for r in results]
        sel_idx = st.selectbox("Select file to inspect:", range(len(results)), format_func=lambda i: file_names[i])
        sel_res = results[sel_idx]
        
        # Renaming Feature
        new_name = st.text_input("Filename", value=sel_res['filename'], key=f"rename_{sel_idx}")
        if new_name != sel_res['filename']:
            sel_res['filename'] = new_name
            results[sel_idx]['filename'] = new_name
            st.rerun()
            
        # Detailed View
        c1, c2 = st.columns(2)
        
        # Detailed View
        c1, c2 = st.columns(2)
        
        # Column 1: Original
        with c1:
            st.markdown("#### **Original**")
            st.video(sel_res['original_data'])

        # Column 2: Processed
        with c2:
            st.markdown("#### **Processed**")
            st.video(sel_res['data'])
            st.download_button("⬇️ Download", sel_res['data'], sel_res['filename'], mime=f"video/{sel_res['ext'][1:]}", use_container_width=True)

        # Deep Analysis Tab (Standardized)
        st.subheader("🔬 Deep Analysis")
        an_t1, an_t2, an_t3 = st.tabs(["📊 Video Properties", "Compare Specs", "🌈 Color Grading"])
        
        with an_t1:
             col1, col2 = st.columns(2)
             with col1:
                 om = sel_res['orig_meta']
                 st.markdown("**Original Specs**")
                 st.write(f"**Resolution**: {om['width']}x{om['height']} px")
                 st.write(f"**Duration**: {om['duration']:.2f}s")
                 st.write(f"**Frame Rate**: {om['fps']:.2f} fps")
                 
             with col2:
                 nm = sel_res['new_meta']
                 st.markdown("**Processed Specs**")
                 st.write(f"**Resolution**: {nm['width']}x{nm['height']} px")
                 st.write(f"**Duration**: {nm['duration']:.2f}s")
                 st.write(f"**Frame Rate**: {nm['fps']:.2f} fps")
                 st.write(f"**Size**: {len(sel_res['data'])/1024/1024:.2f} MB")
                 
        with an_t2:
             # Comparison Table
             st.markdown("**Spec Comparison**")
             comp_data = {
                 "Metric": ["Width", "Height", "FPS", "Duration (s)"],
                 "Original": [om['width'], om['height'], round(om['fps'], 2), round(om['duration'], 2)],
                 "Processed": [nm['width'], nm['height'], round(nm['fps'], 2), round(nm['duration'], 2)]
             }
             st.dataframe(comp_data, hide_index=True, use_container_width=True)
             
        with an_t3:
            st.info("Analysis of the video's color distribution (Keyframe at 50%)")
            c1, c2 = st.columns(2)
            with c1:
                if sel_res.get('hist_orig') is not None:
                    h_chart = alt.Chart(sel_res['hist_orig']).mark_area(opacity=0.5, interpolate='monotone').encode(
                        x=alt.X('Intensity', title=None),
                        y=alt.Y('Count', title=None, axis=None),
                        color=alt.Color('Channel', scale=alt.Scale(domain=['Red', 'Green', 'Blue'], range=['#ef4444', '#22c55e', '#3b82f6']))
                    ).properties(height=200, title="Original Color Profile")
                    st.altair_chart(h_chart, use_container_width=True)
            with c2:
                if sel_res.get('hist_proc') is not None:
                    h_chart = alt.Chart(sel_res['hist_proc']).mark_area(opacity=0.5, interpolate='monotone').encode(
                        x=alt.X('Intensity', title=None),
                        y=alt.Y('Count', title=None, axis=None),
                        color=alt.Color('Channel', scale=alt.Scale(domain=['Red', 'Green', 'Blue'], range=['#ef4444', '#22c55e', '#3b82f6']))
                    ).properties(height=200, title="Processed Color Profile")
                    st.altair_chart(h_chart, use_container_width=True)

        # Bulk Download
        if len(results) > 1:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    zf.writestr(r['filename'], r['data'])
            st.download_button("📦 Download All (ZIP)", zip_buf.getvalue(), "processed_videos.zip", "application/zip", type="primary")
    
    else:
        st.info("👆 Please upload video files to get started")
        
        st.markdown("""
        ### 📋 How to Use Video Studio Pro
        
        #### **Step 1: Upload Video Files**
        - Click the file uploader above
        - Supports: MP4, MOV, AVI, MKV
        - Multiple files can be uploaded at once
        
        #### **Step 2: Select Processing Mode**
        Choose from three powerful modes:
        
        **Standard Processing:**
        - Resize video to HD/Full HD/SD presets
        - Change frame rate (24/30/60 fps)
        - Auto-Enhance quality with CLAHE and sharpening
        
        **Creative Tools (GIF/Audio):**
        - 🎞️ Convert to GIF: Create optimized animated GIFs
        - 🔇 Remove Audio: Strip audio tracks for visual-only files
        
        **Compression:**
        - Reduce file size while maintaining quality
        - Uses CRF 28 for optimal compression
        
        #### **Step 3: Use Preview Mode**
        - ✅ Enable "Preview Mode (First 5s)" to test settings quickly
        - Process only the first 5 seconds before committing to full video
        - Saves time when experimenting with settings
        
        #### **Step 4: Process & Review**
        - Click "🚀 Process Videos"
        - Watch processed videos in the player
        - Review metadata changes (resolution, FPS, duration, file size)
        - Download individual files or all as ZIP
        
        ### 💡 Best Practices
        - Use **Preview Mode** first to test your settings
        - **720p at 30fps** is ideal for most applications
        - **GIF mode** works best for short clips (under 10 seconds)
        - **Remove Audio** for silent videos to reduce file size
        """)

if __name__ == "__main__":
    render()
