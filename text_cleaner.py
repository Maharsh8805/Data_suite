import streamlit as st
import re
import json
import html
from pathlib import Path
import io
import difflib
import pandas as pd
import altair as alt

# Try to import pypdf (optional dependency)
try:
    from pypdf import PdfReader  # type: ignore[import]
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PdfReader = None  # type: ignore[misc,assignment]

# Set page configuration (only when running this file directly)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Text Data Cleaner",
        page_icon="📝",
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
    
    /* Tool Switcher styling */
    div[data-testid="stSelectbox"] > div > div {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #5a5a5a;
        box-shadow: 0 0 0 2px rgba(90, 90, 90, 0.1);
    }
    
    /* Checkbox & Radio styling */
    .stCheckbox label, .stRadio label {
        color: #444;
        font-weight: 500;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #666;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Success message */
    .stSuccess {
        background-color: rgba(209, 250, 229, 0.5);
        border: 1px solid #10b981;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Cleaning Functions -----------------

def strip_html_tags(text):
    """Remove HTML tags from text"""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return text

def remove_special_characters(text, keep_basic=True):
    """Remove special characters while optionally keeping basic punctuation"""
    if keep_basic:
        # Keep letters, numbers, basic punctuation, and whitespace
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"]', '', text)
    else:
        # Keep only letters, numbers, and whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def normalize_case(text, mode='lowercase'):
    """Normalize text case"""
    if mode == 'lowercase':
        return text.lower()
    elif mode == 'uppercase':
        return text.upper()
    elif mode == 'title':
        return text.title()
    return text

def remove_extra_spaces(text):
    """Remove extra spaces and normalize whitespace"""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    return text.strip()

def remove_extra_linebreaks(text):
    """Remove excessive line breaks"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def remove_non_utf8(text):
    """Remove or replace non-UTF-8 characters"""
    return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

def replace_special_whitespace(text):
    """Replace non-breaking spaces and other special whitespace"""
    # Common unicodes
    replacements = {
        '\u00a0': ' ', '\xa0': ' ', '\u2000': ' ', '\u2001': ' ',
        '\u2002': ' ', '\u2003': ' ', '\u2004': ' ', '\u2005': ' ',
        '\u2006': ' ', '\u2007': ' ', '\u2008': ' ', '\u2009': ' ',
        '\u200a': ' ', '\u202f': ' ', '\u205f': ' ', '\u3000': ' ',
        '\t': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def replace_escape_characters(text):
    """Replace common escape characters"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\x00', '').replace('\f', '\n').replace('\v', '\n')
    return text

def remove_urls(text):
    """Remove URLs from text"""
    # Regex to capture http/https/ftp URLs
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

def remove_emails(text):
    """Remove Email addresses"""
    return re.sub(r'\S+@\S+\.\S+', '', text)

def remove_phone_numbers(text):
    """Remove Phone numbers (Global Support: +1, +91, +44, etc.)"""
    # Robust pattern for global phonenumbers
    # Matches: +1-234-567-8900, +91 9876543210, 020 1234 5678, etc.
    # regex matches: optional +code, then groups of digits separated by space/dash/dot
    phone_pattern = r'((\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4})'
    return re.sub(phone_pattern, '', text)

def remove_emojis(text):
    """Remove Emojis (Unicode range based)"""
    # High surrogate range for most emojis
    return re.sub(r'[\U00010000-\U0010ffff]', '', text)

def generate_stats(text):
    """Generate comprehensive text statistics"""
    if not text:
        return {'chars': 0, 'words': 0, 'lines': 0, 'sentences': 0}
    
    return {
        'chars': len(text),
        'words': len(text.split()),
        'lines': len(text.splitlines()) if text else 0,
        'sentences': text.count('.') + text.count('!') + text.count('?')
    }

def generate_diff_html(original, cleaned):
    """Generate HTML diff with red/green highlighting"""
    d = difflib.Differ()
    # Safety check for None inputs
    original = original if original else ""
    cleaned = cleaned if cleaned else ""
    diff = list(d.compare(original.splitlines(), cleaned.splitlines()))
    
    html_content = '<div style="font-family: monospace; white-space: pre-wrap; background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; max-height: 500px; overflow-y: auto;">'
    
    for line in diff:
        code = line[:2]
        text = html.escape(line[2:])
        if code == '- ':
            html_content += f'<div style="background-color: #ffeef0; color: #b31d28; display: block;">- {text}</div>'
        elif code == '+ ':
            html_content += f'<div style="background-color: #e6ffed; color: #22863a; display: block;">+ {text}</div>'
        elif code == '  ':
            html_content += f'<div style="color: #24292e; display: block;">  {text}</div>'
            
    html_content += '</div>'
    return html_content

def clean_text(text, options):
    """Apply all selected cleaning operations"""
    # Validate input
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    
    cleaning_steps = []
    original_stats = generate_stats(text)
    
    # 1. Strip HTML tags
    if options.get('strip_html', False):
        text = strip_html_tags(text)
        cleaning_steps.append("✓ Stripped HTML tags")
    
    # 2. Remove URLs
    if options.get('remove_urls', False):
        text = remove_urls(text)
        cleaning_steps.append("✓ Removed URLs")
        
    # 3. Remove Emails
    if options.get('remove_emails', False):
        text = remove_emails(text)
        cleaning_steps.append("✓ Removed Email addresses")
        
    # 4. Remove Phone Numbers
    if options.get('remove_phone', False):
        text = remove_phone_numbers(text)
        cleaning_steps.append("✓ Removed Phone numbers")

    # 5. Remove Emojis
    if options.get('remove_emojis', False):
        text = remove_emojis(text)
        cleaning_steps.append("✓ Removed Emojis")
    
    # 6. Remove special characters
    if options.get('remove_special_chars', False):
        text = remove_special_characters(text, options.get('keep_basic_punctuation', True))
        cleaning_steps.append("✓ Removed special characters")
    
    # 7. Unicode & Whitespace Normalization (Before case/space cleanup)
    if options.get('remove_non_utf8', False):
        text = remove_non_utf8(text)
        cleaning_steps.append("✓ Removed non-UTF-8 characters")
        
    if options.get('replace_whitespace', False):
        text = replace_special_whitespace(text)
        cleaning_steps.append("✓ Replaced non-breaking spaces")
        
    if options.get('replace_escape', False):
        text = replace_escape_characters(text)
        cleaning_steps.append("✓ Replaced escape characters")
    
    # 8. Normalize case
    if options.get('normalize_case', False):
        text = normalize_case(text, options.get('case_mode', 'lowercase'))
        cleaning_steps.append(f"✓ Normalized case to {options.get('case_mode', 'lowercase')}")
    
    # 9. Clean up Spaces & Lines (Last for cleanest output)
    if options.get('remove_spaces', False):
        text = remove_extra_spaces(text)
        cleaning_steps.append("✓ Removed extra spaces")
    
    if options.get('remove_linebreaks', False):
        text = remove_extra_linebreaks(text)
        cleaning_steps.append("✓ Removed excessive line breaks")
    
    final_stats = generate_stats(text)
    
    return text, cleaning_steps, original_stats, final_stats

def read_file_content(uploaded_file):
    """Read content from uploaded file based on type"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    try:
        content = ""
        file_type = "unknown"
        
        if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.csv']:
            # Robust Text Loading
            try:
                content = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('latin-1')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode('cp1252', errors='ignore')
            file_type = 'text'
        
        elif file_extension == '.json':
            try:
                # Robust JSON Loading
                try:
                    raw = uploaded_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    raw = uploaded_file.read().decode('latin-1')
                    
                json_data = json.loads(raw)
                content = extract_text_from_json(json_data)
                file_type = 'json'
            except json.JSONDecodeError:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                file_type = 'text' # Fallback to raw text
        
        elif file_extension == '.pdf':
            if PDF_AVAILABLE:
                pdf_reader = PdfReader(uploaded_file)
                text_list = []
                for page in pdf_reader.pages:
                    text_list.append(page.extract_text())
                content = "\n".join(text_list)
                file_type = 'pdf'
            else:
                st.error("PDF support is missing. Please install 'pypdf'.")
                return None, None
        
        else:
            # Fallback for unknown extensions
            try:
                content = uploaded_file.read().decode('utf-8')
            except:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('latin-1', errors='ignore')
            file_type = 'text'
            
        return content, file_type
    
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
        return None, None

def extract_text_from_json(data, depth=0):
    """Recursively extract text from JSON structure"""
    texts = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str): texts.append(value)
            elif isinstance(value, (dict, list)): texts.append(extract_text_from_json(value, depth + 1))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str): texts.append(item)
            elif isinstance(item, (dict, list)): texts.append(extract_text_from_json(item, depth + 1))
    elif isinstance(data, str):
        texts.append(data)
    return '\n'.join(filter(None, texts))

def render():
    # Navigation is handled by the Global Sidebar in main_app.py

    st.markdown("---")
    st.title("📝 Text Data Cleaner")
    st.write("Professional text sanitization and normalization tool. Upload files or paste text below.")

    # Input method
    input_method = st.radio("Choose input method:", ["Upload Files (TXT, JSON, PDF)", "Paste Text"], horizontal=True)

    text_to_clean = None

    if "Upload" in input_method:
        uploaded_files = st.file_uploader(
            "Upload text files",
            type=['txt', 'json', 'pdf'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if not PDF_AVAILABLE and any(f.name.endswith('.pdf') for f in uploaded_files):
                 st.warning("⚠️ 'pypdf' is not installed. PDF files will not be processed correctly.")

            all_texts = []
            for uploaded_file in uploaded_files:
                content, file_type = read_file_content(uploaded_file)
                if content:
                    all_texts.append(content)

            if all_texts:
                text_to_clean = '\n\n'.join(all_texts)
                st.success(f"✅ Successfully loaded {len(all_texts)} file(s)")

    else:
        text_input = st.text_area(
            "Paste your text here:",
            height=250,
            placeholder="Enter text to clean..."
        )
        if st.button("📥 Load Text", type="secondary"):
            if text_input:
                text_to_clean = text_input
                st.success("Text loaded!")
        if text_input:
             text_to_clean = text_input

    if text_to_clean:
        st.markdown("### 🧹 Cleaning Options")
        
        # Organized cleaning options in tabs or columns? Columns are better for checklist feel.
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### **Content Removal**")
            strip_html = st.checkbox("Strip HTML tags", value=True, help="Removes all HTML tags like <div>, <br>, etc.")
            remove_urls_opt = st.checkbox("Remove URLs", value=False, help="Removes http/https links")
            remove_emails_opt = st.checkbox("Remove Emails", value=False, help="Removes email addresses")
            remove_phone_opt = st.checkbox("Remove Phone Numbers", value=False, help="Removes phone number patterns")
            remove_emojis_opt = st.checkbox("Remove Emojis", value=False, help="Removes unicode emojis")

        with col2:
            st.markdown("#### **Characters & Case**")
            remove_special_chars = st.checkbox("Remove Special Characters", value=True, help="Removes non-alphanumeric characters")
            keep_basic_punctuation = st.checkbox("Keep Basic Punctuation (.,!?;:-)", value=True, disabled=not remove_special_chars)
            normalize_case_opt = st.checkbox("Normalize Case", value=True)
            case_mode = st.selectbox("Case Mode", ['lowercase', 'uppercase', 'title'], index=2, disabled=not normalize_case_opt)

        with col3:
            st.markdown("#### **Whitespace & Format**")
            remove_spaces = st.checkbox("Remove Extra Spaces", value=True, help="Reduces multiple spaces to single space")
            remove_linebreaks = st.checkbox("Remove Extra Line Breaks", value=True, help="Removes 3+ consecutive line breaks")
            remove_non_utf8_opt = st.checkbox("Fix Encoding (UTF-8 Only)", value=True, help="Removes non-printable/bad characters")
            replace_whitespace = st.checkbox("Fix Non-Breaking Spaces", value=True, help="Converts nbsp to normal space")
            replace_escape = st.checkbox("Fix Escape Characters", value=True, help="Fixes newline/tab escapes")

        if st.button("🚀 Clean Text Now", type="primary", width="stretch"):
            cleaning_options = {
                'strip_html': strip_html,
                'remove_urls': remove_urls_opt,
                'remove_emails': remove_emails_opt,
                'remove_phone': remove_phone_opt,
                'remove_emojis': remove_emojis_opt,
                'remove_special_chars': remove_special_chars,
                'keep_basic_punctuation': keep_basic_punctuation,
                'normalize_case': normalize_case_opt,
                'case_mode': case_mode,
                'remove_spaces': remove_spaces,
                'remove_linebreaks': remove_linebreaks,
                'remove_non_utf8': remove_non_utf8_opt,
                'replace_whitespace': replace_whitespace,
                'replace_escape': replace_escape
            }

            with st.spinner("Processing..."):
                cleaned_text, steps, original_stats, final_stats = clean_text(text_to_clean, cleaning_options)
                
                # Save to session (optional, but good for reruns)
                st.session_state['cleaned_result'] = {
                    'text': cleaned_text,
                    'steps': steps,
                    'orig_stats': original_stats,
                    'final_stats': final_stats
                }

    # Display Results if available
    if 'cleaned_result' in st.session_state:
        res = st.session_state['cleaned_result']
        
        st.markdown("---")
        
        # Header with Reset Button
        r1, r2 = st.columns([4, 1])
        with r1:
            st.subheader("📊 Results & Statistics")
        with r2:
            if st.button("🔄 Start Over", type="secondary", width="stretch"):
                del st.session_state['cleaned_result']
                st.rerun()
        
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Characters", f"{res['final_stats']['chars']:,}", delta=f"{res['final_stats']['chars'] - res['orig_stats']['chars']:,}")
        m2.metric("Words", f"{res['final_stats']['words']:,}", delta=f"{res['final_stats']['words'] - res['orig_stats']['words']:,}")
        m3.metric("Lines", f"{res['final_stats']['lines']:,}", delta=f"{res['final_stats']['lines'] - res['orig_stats']['lines']:,}")
        m4.metric("Sentences (Approx)", f"{res['final_stats']['sentences']:,}")

        # Interactive Comparison Chart
        st.markdown("#### 📈 Statistics Comparison")
        
        stats_data = pd.DataFrame({
            'Metric': ['Characters', 'Words', 'Lines', 'Sentences'],
            'Original': [res['orig_stats']['chars'], res['orig_stats']['words'], res['orig_stats']['lines'], res['orig_stats']['sentences']],
            'Cleaned': [res['final_stats']['chars'], res['final_stats']['words'], res['final_stats']['lines'], res['final_stats']['sentences']]
        })
        
        # Melt for Altair
        stats_melted = stats_data.melt('Metric', var_name='Version', value_name='Count')
        
        chart = alt.Chart(stats_melted).mark_bar().encode(
            x=alt.X('Metric', sort=['Characters', 'Words', 'Lines', 'Sentences']),
            y=alt.Y('Count'),
            color=alt.Color('Version', scale=alt.Scale(domain=['Original', 'Cleaned'], range=['#94a3b8', '#10b981'])),
            column=alt.Column('Metric', header=alt.Header(title=None, labelFontSize=12)),
            tooltip=['Metric', 'Version', 'Count']
        ).properties(width=150, height=200).interactive()
        
        st.altair_chart(chart)

        with st.expander("🛠️ View Applied Cleaning Steps"):
            for step in res['steps']:
                st.write(step)
        
        st.subheader("✨ Cleaned Output")
        
        tab_preview, tab_diff, tab_full = st.tabs(["👁️ Preview", "🔍 Visual Diff", "📝 Full Text"])
        
        with tab_preview:
            st.text_area("First 1000 characters", res['text'][:1000], height=300)
            
        with tab_diff:
            st.markdown("#### Comparison (Red = Removed, Green = Added)")
            diff_html = generate_diff_html(text_to_clean, res['text'])
            st.markdown(diff_html, unsafe_allow_html=True)
            
        with tab_full:
            st.text_area("Full Output", res['text'], height=500, key="full_text_output")
            
        # Download Section
        st.subheader("📥 Download")
        
        # Renaming Feature
        base_name = st.text_input("Base Filename", value="cleaned_text")
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 Download as Text File (.txt)",
                data=res['text'],
                file_name=f"{base_name}.txt",
                mime="text/plain",
                type="primary",
                width="stretch"
            )
        with c2:
            json_dump = json.dumps({
                "cleaned_text": res['text'],
                "stats": res['final_stats'],
                "steps": res['steps']
            }, indent=2)
            st.download_button(
                "📋 Download as JSON (.json)",
                data=json_dump,
                file_name=f"{base_name}.json",
                mime="application/json",
                width="stretch"
            )

    else:
        st.info("👆 Please upload files or paste text to get started")
        
        # Display comprehensive workflow instructions
        st.markdown("""
        ### 📋 How to Use Text Cleaner
        
        Follow these steps to clean and process your text data:
        
        #### **Step 1: Input Text Data**
        - **Option A: Paste Text** - Directly paste text into the text area
        - **Option B: Upload Files** - Upload one or more files
        - **Supported Formats**: TXT, JSON, PDF (automatically extracts text)
        
        #### **Step 2: Configure Cleaning Options**
        Select the cleaning operations you want to apply:
        
        **Content Removal:**
        - ✅ **Strip HTML tags** - Removes all HTML/XML tags
        - ✅ **Remove URLs/Emails/Phones** - Privacy and noise cleanup
        - ✅ **Remove Emojis** - Cleans non-standard characters
        
        **Characters & Case:**
        - ✅ **Remove special characters** - Keeps only alphanumeric and basic punctuation
        - ✅ **Normalize case** - Convert to Lowercase, Uppercase, or Title Case
        
        **Whitespace & Format:**
        - ✅ **Remove extra spaces** - Collapses multiple spaces into one
        - ✅ **Remove Extra Line Breaks** - Removes 3+ consecutive line breaks
        - ✅ **Fix Encoding** - Removes non-UTF-8 characters
        - ✅ **Fix Non-Breaking Spaces** - Converts nbsp to normal space
        - ✅ **Fix Escape Characters** - Fixes newline/tab escapes
        
        #### **Step 3: Process & Verify**
        - Click **"🚀 Clean Text Now"**
        - Use the **🔍 Visual Diff** tab to see exactly what changed (Red = Removed, Green = Added)
        - Check "Results & Statistics" for detailed metrics
        
        #### **Step 4: Download Results**
        - **Download as TXT** - Get the raw cleaned text file
        - **Download as JSON** - Get structured data including statistics and steps
        
        ---
        
        ### 💡 Tips for Best Results
        - Use **"Visual Diff"** to verify intricate cleaning (like regex removals) before downloading.
        - **"Normalize case"** is great for NLP datasets.
        - Start with default settings and adjust if too aggressive.
        
        ### 🎯 Common Use Cases
        - **NLP/Machine Learning**: Prepare text for model training
        - **Web Scraping**: Clean extracted content from websites to remove HTML and links
        - **Data Analysis**: Standardize text for sentiment analysis
        - **Privacy**: Quickly remove emails and phone numbers from documents
        
        ### 📊 Supported Formats
        - **Input**: TXT, JSON, PDF (native text extraction)
        - **Output**: TXT (raw text), JSON (with metadata)
        """)

if __name__ == "__main__":
    render()