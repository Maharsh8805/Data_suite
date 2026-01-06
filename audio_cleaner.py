# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import io
import zipfile
from pathlib import Path
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt

# Try to import pydub and check audio support
try:
    from pydub import AudioSegment
    from pydub.silence import detect_leading_silence, split_on_silence
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False

# Set page configuration
if __name__ == "__main__":
    st.set_page_config(
        page_title="Audio Data Cleaner",
        page_icon="🎵",
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
    
    /* Button styling handling by main_app.py global styles */
    
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
    
    /* Tool Switcher styling */
    div[data-testid="stSelectbox"] > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
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
    
    /* Result card styling */
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

if AUDIO_SUPPORT:
    import ffmpeg_manager
    ffmpeg_manager.configure_pydub()

    # --- Processing Functions ---
    def do_voice_optimization(audio, strict=False):
        """Apply filters for voice"""
        if strict:
            # Telephone band (300-3400Hz)
            return audio.high_pass_filter(300).low_pass_filter(3400)
        else:
            # Standard voice EQ (100-8000Hz)
            return audio.high_pass_filter(100).low_pass_filter(8000)

    def do_trim_silence(audio, thresh=-50.0):
        """Trim start/end silence"""
        start = detect_leading_silence(audio, thresh, chunk_size=5)
        end = detect_leading_silence(audio.reverse(), thresh, chunk_size=5)
        return audio[start:len(audio)-end]

    def do_remove_gaps(audio, thresh=-50.0, min_len=500, keep=250):
        """Remove audio gaps"""
        try:
            chunks = split_on_silence(audio, min_silence_len=min_len, silence_thresh=thresh, keep_silence=keep, seek_step=10)
            if not chunks: return audio
            # Use export/import instead of raw_data to avoid type issues
            combined = AudioSegment.empty()
            for chunk in chunks:
                combined += chunk
            return combined
        except Exception:
            return audio

    def generate_waveform(audio_segment, title="Waveform", color='#4a90e2'):
        """Generate a waveform plot image"""
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Downsample for performance (max 10k points)
        step = max(1, len(samples) // 10000)
        samples = samples[::step]
        
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(samples, color=color, linewidth=0.5, alpha=0.8)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, pad=5, color='#333')
        
        # Make transparent
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf



    def get_audio_properties(audio_segment):
        """Extract technical properties"""
        return {
            "Duration": f"{len(audio_segment)/1000:.2f}s",
            "Channels": f"{audio_segment.channels} ({'Mono' if audio_segment.channels==1 else 'Stereo'})",
            "Sample Rate": f"{audio_segment.frame_rate} Hz",
            "Bit Depth": f"{audio_segment.sample_width * 8}-bit",
            "dBFS": f"{audio_segment.dBFS:.1f} dB"
        }

    def process_audio_file(uploaded_file, options):
        """Process main audio logic"""
        # Validate file
        if uploaded_file is None or uploaded_file.size == 0:
            return {'error': 'Invalid or empty audio file'}
        
        ffmpeg_path, _ = ffmpeg_manager.get_ffmpeg_paths()
        if not ffmpeg_path: 
            return {'error': 'FFmpeg not found'}
        AudioSegment.converter = ffmpeg_path
        
        # Save temp
        ext = Path(uploaded_file.name).suffix.lower()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            audio = AudioSegment.from_file(tmp_path)
            if len(audio) == 0:
                return {'error': 'Audio file is empty or invalid'}
            orig_audio = audio # Keep reference
            orig_duration = len(audio) / 1000.0
            
            steps = []
            
            # 0. Pre-Normalize
            audio = audio.apply_gain(-20.0 - audio.dBFS)
            steps.append("Normalized to -20dB")
            
            # 1. Voice Optimization
            if options.get('optimize'):
                audio = do_voice_optimization(audio, options.get('strict'))
                steps.append(f"Voice EQ ({'Strict' if options.get('strict') else 'Standard'})")
                
            # 2. Mono
            if options.get('mono') and audio.channels > 1:
                audio = audio.set_channels(1)
                steps.append("Converted to Mono")
                
            # 3. Silence Trimming
            thresh = options.get('silence_thresh', -40)
            if options.get('auto_thresh'):
                # Heuristic: Ave - 10dB (Standard) or -6dB (Strict)
                thresh = audio.dBFS - (6 if options.get('strict') else 10)
                thresh = max(min(thresh, -15), -50)
                steps.append(f"Auto-Threshold: {thresh:.1f}dB")
                
            if options.get('trim'):
                prev_len = len(audio)
                audio = do_trim_silence(audio, thresh)
                if len(audio) < prev_len: steps.append("Trimmed Silence")
                
            if options.get('remove_gaps'):
                prev_len = len(audio)
                audio = do_remove_gaps(audio, thresh, options.get('min_gap', 500))
                if len(audio) < prev_len: steps.append("Removed internal gaps")
                
            # 4. Final Params
            if options.get('format'):
                audio = audio.set_frame_rate(options['rate'])
                audio = audio.set_sample_width(options['width'])
                steps.append(f"Format: {options['rate']}Hz, {options['width']*8}-bit")

            new_duration = len(audio) / 1000.0
            
            # Export
            out_buf = io.BytesIO()
            audio.export(out_buf, format=options['out_fmt'])
            out_buf.seek(0)
            
            # Waveforms (if requested, costly but premium)
            # We assume we want them for the results view
            
            return {
                'filename': uploaded_file.name,
                'orig_audio': orig_audio, # Careful with memory, but needed for visual diff? 
                # actually passing AudioSegment objects around in session state is fine for small files.
                'proc_audio': audio,
                'data': out_buf.getvalue(),
                'steps': steps,
                'orig_dur': orig_duration,
                'new_dur': new_duration,
                'thresh_used': thresh,
                'orig_props': get_audio_properties(orig_audio),
                'new_props': get_audio_properties(audio)
            }
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)


def render():
    # Navigation is handled by the Global Sidebar in main_app.py

    st.markdown("---")
    st.title("🎵 Audio Processor Pro")
    st.write("Professional voice optimization and audio standardization.")

    if not AUDIO_SUPPORT:
        st.error("⚠️ Audio libraries missing. Please install pydub and ffmpeg.")
        return

    # Upload
    uploaded_files = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'm4a', 'flac', 'ogg'], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"📂 {len(uploaded_files)} files loaded")
        
        # Settings Layout
        st.subheader("🎛️ Processing Settings")
        
        tab_clean, tab_fmt, tab_out = st.tabs(["🎙️ Voice & Noise", "⚙️ Format", "💾 Output"])
        
        with tab_clean:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### **Optimization**")
                optimize = st.checkbox("🎙️ Optimize for Voice", value=True, help="Applies EQ to focus on speech frequencies.")
                strict = st.checkbox("🚧 Strict Isolation", value=False, help="Aggressive filtering (Telephone band).")
            with c2:
                st.markdown("#### **Silence Removal**")
                auto_silence = st.checkbox("✂️ Auto-Remove Silence", value=True)
                trim = st.checkbox("Trim Ends", value=True, disabled=not auto_silence)
                gaps = st.checkbox("Remove Gaps", value=True, disabled=not auto_silence)
                
        with tab_fmt:
            c1, c2 = st.columns(2)
            with c1:
                convert_fmt = st.checkbox("Convert Format", value=True)
                rate = st.selectbox("Sample Rate", [16000, 44100, 48000], index=0, disabled=not convert_fmt)
            with c2:
                mono = st.checkbox("Convert to Mono", value=True)
                width = st.selectbox("Bit Depth", ["16-bit", "24-bit"], index=0, disabled=not convert_fmt)
                width_bytes = 2 if width == "16-bit" else 3

        with tab_out:
            c1, c2 = st.columns(2)
            with c1:
                out_fmt = st.selectbox("Output Format", ["wav", "mp3", "flac"])
            with c2:
                rename = st.checkbox("Rename Files", value=True)
                prefix = st.text_input("Prefix", "voice_clean", disabled=not rename)

        if st.button("🚀 Process Audio", type="primary", width="stretch"):
            opts = {
                'optimize': optimize, 'strict': strict,
                'auto_thresh': auto_silence, 'trim': trim, 'remove_gaps': gaps,
                'format': convert_fmt, 'rate': rate, 'width': width_bytes, 'mono': mono,
                'out_fmt': out_fmt
            }
            
            with st.spinner("Processing audio..."):
                results = []
                for idx, f in enumerate(uploaded_files):
                    res = process_audio_file(f, opts)
                    if 'error' not in res:
                        if rename:
                            res['filename'] = f"{prefix}_{idx+1:03d}.{out_fmt}"
                        results.append(res)
                
                st.session_state['audio_results'] = results
                st.rerun()

    # Results View
    if 'audio_results' in st.session_state:
        results = st.session_state['audio_results']
        st.markdown("---")
        
        # Header + Reset
        r1, r2 = st.columns([4, 1])
        with r1: st.subheader(f"📊 Results ({len(results)} files)")
        with r2: 
            if st.button("🔄 Start Over", type="secondary"):
                del st.session_state['audio_results']
                st.rerun()

        # Visual Comparison Section (Top Notch Feature)
        st.subheader("🌊 Waveform Comparison")
        
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
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.image(generate_waveform(sel_res['orig_audio'], color='#999'), width='stretch')
            st.write(f"⏱️ {sel_res['orig_dur']:.2f}s")
            # We can't play original safely if it wasn't saved? 
            # Actually we didn't save original bytes in result dict to save mem, but we have orig_audio object
            # We can export it to bytes on the fly for playback
            buf = io.BytesIO()
            sel_res['orig_audio'].export(buf, format="wav") # preview as wav
            st.audio(buf.getvalue(), format="audio/wav")
            
        with c2:
            st.markdown("**Processed**")
            st.image(generate_waveform(sel_res['proc_audio'], color='#2ecc71'), width='stretch')
            st.write(f"⏱️ {sel_res['new_dur']:.2f}s")
            st.audio(sel_res['data'], format=f"audio/{out_fmt}")
            
        with st.expander("🛠️ Applied Steps"):
            for s in sel_res['steps']: st.write(f"- {s}")

        # Advanced Analysis
        st.subheader("🔬 Deep Analysis")
        # Removed Spectrogram as per user request
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original File Details**")
            st.dataframe(pd.DataFrame(list(sel_res['orig_props'].items()), columns=['Property', 'Value']), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Processed File Details**")
            st.dataframe(pd.DataFrame(list(sel_res['new_props'].items()), columns=['Property', 'Value']), hide_index=True, use_container_width=True)

        st.markdown("---")
        
        # Download
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
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
        
        st.download_button("📦 Download All (ZIP)", zip_buf.getvalue(), "cleaned_audio.zip", "application/zip", type="primary", width="stretch")
    
    else:
        st.info("👆 Please upload audio files to get started")
        
        st.markdown("""
        ### 📋 How to Use Audio Processor Pro
        
        #### **Step 1: Upload Audio Files**
        - Click the file uploader above
        - Supports: WAV, MP3, M4A, FLAC, OGG
        - Multiple files can be uploaded at once
        
        #### **Step 2: Choose Processing Mode**
        Select from three powerful modes:
        
        **🎙️ Voice & Noise:**
        - Optimize for Voice: Removes background noise and focuses on speech frequencies
        - Strict Isolation: Aggressive filtering for single-speaker scenarios
        - Strict Isolation: Aggressive filtering for single-speaker scenarios
        - Auto-Remove Silence: Intelligent detection and removal of silent parts
        - Trim Ends: Removes silence from start and end of audio
        - Remove internal Gaps: Removes long pauses between words/segments
        
        **⚙️ Format:**
        - Convert sample rate (16kHz recommended for speech)
        - Change bit depth (16-bit standard)
        - Convert to mono for consistency
        
        **💾 Output:**
        - Choose output format (WAV, MP3, FLAC)
        - Rename files with custom prefix
        
        #### **Step 3: Process & Review**
        - Click "🚀 Process Audio"
        - View waveform comparison (Before vs After)
        - Listen to original and processed audio
        - Review applied processing steps
        
        #### **Step 4: Download**
        - Download all files as ZIP
        
        ### 💡 Best Practices
        - Use **16kHz, 16-bit, mono WAV** for speech recognition
        - Enable **Auto-Remove Silence** for cleaner datasets
        - Use **Strict Isolation** for noisy environments
        """)

if __name__ == "__main__":
    render()
