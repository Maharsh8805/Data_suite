import os
import sys
import shutil
import zipfile
import requests
import streamlit as st
from pathlib import Path
import io

# Define local bin directory within the project
LOCAL_BIN_DIR = Path("bin").absolute()
FFMPEG_EXE = LOCAL_BIN_DIR / "ffmpeg.exe"
FFPROBE_EXE = LOCAL_BIN_DIR / "ffprobe.exe"

def check_local_ffmpeg():
    """Check if valid ffmpeg binaries exist in local bin"""
    if FFMPEG_EXE.exists() and FFPROBE_EXE.exists():
        return str(FFMPEG_EXE), str(FFPROBE_EXE)
    return None, None

def download_ffmpeg():
    """Download and setup FFmpeg locally"""
    # URL for a reliable Windows static build (Gyan.dev essentials)
    # Using a specific release to ensure stability, or latest-essentials
    URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    
    try:
        LOCAL_BIN_DIR.mkdir(exist_ok=True)
        
        with st.status("🛠️ Setting up audio processing tools (FFmpeg)... this happens only once.", expanded=True) as status:
            status.write("Downloading FFmpeg binaries (approx. 30MB)...")
            response = requests.get(URL, stream=True)
            response.raise_for_status()
            
            status.write("Extracting files...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the exe files inside the zip (they are usually in a subdir)
                for info in z.infolist():
                    if info.filename.endswith("ffmpeg.exe"):
                        with open(FFMPEG_EXE, "wb") as f_out:
                            f_out.write(z.read(info.filename))
                    elif info.filename.endswith("ffprobe.exe"):
                        with open(FFPROBE_EXE, "wb") as f_out:
                            f_out.write(z.read(info.filename))
            
            if FFMPEG_EXE.exists() and FFPROBE_EXE.exists():
                status.update(label="✅ Setup complete!", state="complete", expanded=False)
                return str(FFMPEG_EXE), str(FFPROBE_EXE)
            else:
                status.update(label="❌ Extracted files not found!", state="error")
                st.error("Could not find ffmpeg.exe or ffprobe.exe in the downloaded archive.")
                return None, None
                
    except Exception as e:
        st.error(f"Failed to auto-install FFmpeg: {str(e)}")
        return None, None

def get_ffmpeg_paths():
    """Get paths to FFMPEG and FFPROBE, installing if necessary"""
    # 1. Check local bin first (priority)
    ffmpeg, ffprobe = check_local_ffmpeg()
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe

    # 2. Check system path (fallback)
    sys_ffmpeg = shutil.which("ffmpeg")
    sys_ffprobe = shutil.which("ffprobe")
    
    if sys_ffmpeg and sys_ffprobe:
        # Verify they work/exist? shutil.which returns path
        return sys_ffmpeg, sys_ffprobe
        
    # 3. If mostly missing, try to fix the broken C:\ffmpeg case the user had
    # User had ffmpeg but not ffprobe in C:\ffmpeg
    # If we are here, we either have nothing, or partial.
    # Let's auto-install to local bin to be safe and consistent
    return download_ffmpeg()

def configure_pydub():
    """Configure pydub to use our resolved paths"""
    ffmpeg_path, ffprobe_path = get_ffmpeg_paths()
    
    if ffmpeg_path and ffprobe_path:
        # Add to PATH so pydub subprocess calls find them
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
            
        # Also strictly set pydub variables if possible (depends on pydub version)
        from pydub import AudioSegment
        AudioSegment.converter = ffmpeg_path
        
        return True
    return False
