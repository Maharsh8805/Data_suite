import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import altair as alt

# Set page configuration (only when running this file directly)
if __name__ == "__main__":
    st.set_page_config(
        page_title="CSV Data Cleaner",
        page_icon="📊",
        layout="wide"
    )

# Custom CSS for background and styling
st.markdown("""
    <style>
    /* Main background gradient */
    /* Content area styling - Transparent for Glassmorphism inheritance */
    .main .block-container {
        border-radius: 15px;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #333333;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: rgba(224, 224, 224, 0.1);
        padding: 1rem;
        border-radius: 10px;
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
    
    /* Checkbox styling */
    .stCheckbox {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #5a5a5a 0%, #3e3e3e 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-bottom: 3px solid #5a5a5a;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def clean_data(df, options):
    """Apply data cleaning operations based on selected options"""
    df_cleaned = df.copy()
    cleaning_report = []
    
    # Strip whitespace from headers
    if options.get('strip_headers', False):
        original_cols = df_cleaned.columns.tolist()
        df_cleaned.columns = df_cleaned.columns.str.strip()
        cleaned_cols = df_cleaned.columns.tolist()
        if original_cols != cleaned_cols:
            cleaning_report.append("✓ Stripped whitespace from column headers")
    
    # Strip whitespace from string columns
    if options.get('strip_strings', False):
        string_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in string_cols:
            df_cleaned[col] = df_cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        if len(string_cols) > 0:
            cleaning_report.append(f"✓ Stripped whitespace from {len(string_cols)} string column(s)")
    
    # Convert date columns to datetime
    if options.get('convert_dates', False):
        date_cols = []
        # Determine dayfirst based on user preference
        day_first = False # Default US-centric
        fmt_pref = options.get('date_format', 'Auto')
        
        if "Global" in fmt_pref:
            day_first = True
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                try:
                    # Try to convert to datetime (suppress warning for mixed formats)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        converted = pd.to_datetime(df_cleaned[col], errors='coerce', dayfirst=day_first)
                    # If more than 50% successfully converted, assume it's a date column
                    if len(df_cleaned) > 0 and converted.notna().sum() / len(df_cleaned) > 0.5:
                        df_cleaned[col] = converted
                        date_cols.append(col)
                except:
                    pass
        if date_cols:
            cleaning_report.append(f"✓ Converted {len(date_cols)} column(s) to datetime: {', '.join(date_cols)} (DayFirst={day_first})")
    
    # Lowercase all text entries in object columns
    if options.get('lowercase_text', False):
        text_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in text_cols:
            df_cleaned[col] = df_cleaned[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
        if len(text_cols) > 0:
            cleaning_report.append(f"✓ Converted text to lowercase in {len(text_cols)} column(s)")
    
    # Remove duplicate rows
    if options.get('remove_duplicates', False):
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)
        if duplicates_removed > 0:
            cleaning_report.append(f"✓ Removed {duplicates_removed} duplicate row(s)")
        else:
            cleaning_report.append("✓ No duplicate rows found")
    
    # Remove outliers
    if options.get('remove_outliers', False):
        method = options.get('outlier_method', 'z_score')
        threshold = options.get('outlier_threshold', 3.0)
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        initial_rows = len(df_cleaned)
        
        if method == 'z_score':
            # Z-Score method
            for col in numeric_cols:
                std_val = df_cleaned[col].std()
                if std_val > 0:  # Avoid division by zero for constant columns
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / std_val)
                    df_cleaned = df_cleaned[z_scores < threshold]

        elif method == 'iqr':
            # IQR method
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid filtering if IQR is zero (constant column)
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                        
        outliers_removed = initial_rows - len(df_cleaned)
        if outliers_removed > 0:
            cleaning_report.append(f"✓ Removed {outliers_removed} outlier(s) using {method.upper()}")
        else:
            cleaning_report.append(f"✓ No outliers found using {method.upper()}")

    return df_cleaned, cleaning_report

def visualize_data(df):
    """Generate simplified visualizations for the dataframe"""
    st.subheader("📊 Data Distribution & Insights")
    
    
    # 1. Key Takeaways (Smart Summary)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("### 🧠 Key Takeaways")
    with col2:
        if st.button("🔄 Start Over", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    row_count = len(df)
    col_count = len(df.columns)
    
    # Generate generic insights
    insights = []
    insights.append(f"The dataset contains **{row_count} rows** and **{col_count} columns**.")
    
    # Check for likely ID columns (all unique)
    unique_cols = [col for col in df.columns if df[col].nunique() == row_count]
    if unique_cols:
        insights.append(f"Columns likely serving as IDs: **{', '.join(unique_cols[:3])}**.")
    
    for note in insights:
        st.write(f"- {note}")
    
    st.divider()

    # 2. Categorical Analysis (Top N)
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    if len(categorical_cols) > 0:
        st.write("### 🏷️ Top Categories")
        selected_cat_col = st.selectbox("Select category to analyze:", categorical_cols)
        
        if selected_cat_col:
            # Get Top 10 counts
            top_counts = df[selected_cat_col].value_counts().head(10).reset_index()
            top_counts.columns = ['Category', 'Count']
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Interactive Altair Chart
                chart = alt.Chart(top_counts).mark_bar().encode(
                    x=alt.X('Count', title='Frequency'),
                    y=alt.Y('Category', sort='-x', title='Category'),
                    color=alt.Color('Count', scale=alt.Scale(scheme='viridis'), legend=None),
                    tooltip=['Category', 'Count']
                ).properties(height=300).interactive()
                st.altair_chart(chart, use_container_width=True)
                
            with col2:
                st.write("**Most Frequent Values:**")
                st.dataframe(top_counts, use_container_width=True, hide_index=True)
                
                # Smart explanation
                if not top_counts.empty:
                    top_name = top_counts.iloc[0]['Category']
                    top_val = top_counts.iloc[0]['Count']
                    pct = (top_val / row_count) * 100
                    st.info(f"💡 **'{top_name}'** is the most common value, appearing in **{pct:.1f}%** of records.")
        st.divider()

    # 3. Numeric Analysis (Distributions & Correlations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("### 📈 Numeric Trends")
        
        # Tabs for numeric sub-views
        num_tab1, num_tab2 = st.tabs(["Distribution", "Correlations"])
        
        with num_tab1:
            selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
            if selected_num_col:
                # Stats
                avg = df[selected_num_col].mean()
                med = df[selected_num_col].median()
                std = df[selected_num_col].std()
                min_v = df[selected_num_col].min()
                max_v = df[selected_num_col].max()
                
                # Interactive Histogram
                base = alt.Chart(df).encode(alt.X(selected_num_col, bin=alt.Bin(maxbins=30), title=selected_num_col))
                hist = base.mark_bar(opacity=0.7).encode(
                    y=alt.Y('count()', title='Frequency'),
                    color=alt.value('#6366f1'),
                    tooltip=[alt.Tooltip(selected_num_col, bin=True), 'count()']
                ).interactive()
                
                # Density Line (Optional, simplified to just rule for mean)
                rule = alt.Chart(pd.DataFrame({'Mean': [avg]})).mark_rule(color='red').encode(x='Mean', size=alt.value(2))
                
                st.altair_chart((hist + rule).properties(height=300), use_container_width=True)
                
                # Smart explanation
                st.info(f"💡 The average value is **{avg:.2f}** (Red Line). Most data points cluster around **{med:.2f}**.")
                
                # Metric Cards
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{avg:.2f}")
                c2.metric("Median", f"{med:.2f}")
                c3.metric("Min", f"{min_v:.2f}")
                c4.metric("Max", f"{max_v:.2f}")

        with num_tab2:
            st.write("Correlation Matrix (Dependencies)")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr().reset_index().melt(id_vars='index')
                corr.columns = ['Var1', 'Var2', 'Correlation']
                
                heatmap = alt.Chart(corr).mark_rect().encode(
                    x='Var1',
                    y='Var2',
                    color=alt.Color('Correlation', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                    tooltip=['Var1', 'Var2', alt.Tooltip('Correlation', format='.2f')]
                ).properties(height=400).interactive()
                
                st.altair_chart(heatmap, use_container_width=True)
                st.caption("Values close to 1.0 mean strong positive relationship. Values close to -1.0 mean strong negative relationship.")
            else:
                st.info("Need at least 2 numeric columns for correlation.")
    else:
        st.info("No numeric columns found for visualization.")

def render():
    # Navigation is handled by the Global Sidebar in main_app.py
    
    st.markdown("---")
    
    st.title("📊 Automatic CSV Data Cleaner")
    st.write("Upload a CSV file to clean and process your data automatically")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the CSV file
        # Robust CSV Loading
        try:
            # Try default UTF-8 first
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            try:
                # Try Latin-1 (common fallback)
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
            except UnicodeDecodeError:
                try:
                    # Try CP1252 (Windows default)
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp1252')
                except Exception as e:
                    st.error(f"⚠️ Error reading CSV file. Please check encoding (Supported: UTF-8, Latin-1, CP1252). Error: {str(e)}")
                    return
        except Exception as e:
             st.error(f"⚠️ Error parsing CSV file: {str(e)}")
             return
            
        if df.empty:
            st.error("⚠️ The uploaded CSV file is empty. Please upload a file with data.")
            return

        # Tabs for better organization
        tab_clean, tab_viz = st.tabs(["🧹 Data Cleaning", "📊 Data Insights"])

        with tab_clean:
            st.subheader("🧹 Data Preprocessing Options")
            st.write("Select preprocessing operations to perform before handling null values:")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### Text & Format")
                strip_headers = st.checkbox("Strip whitespace from headers", value=True)
                strip_strings = st.checkbox("Strip whitespace from text", value=True)
                convert_dates = st.checkbox("Convert date columns", value=True)
                date_format = "Auto"
                if convert_dates:
                     date_format = st.selectbox("Date Format", ["Auto", "Global (DD/MM/YYYY)", "US (MM/DD/YYYY)"], index=0)
                lowercase_text = st.checkbox("Lowercase all text", value=False)

            with col2:
                st.write("### Structure & quality")
                remove_duplicates = st.checkbox("Remove duplicate rows", value=True)


            with col3:
                st.write("### 🔮 Advanced (Outliers)")
                remove_outliers = st.checkbox("Remove Statistical Outliers", value=False, help="Removes rows with values far from the mean.")
                outlier_method = st.selectbox("Method", ["z_score", "iqr"], disabled=not remove_outliers)
                outlier_threshold = st.slider("Threshold (Sigma)", 2.0, 5.0, 3.0, disabled=not remove_outliers or outlier_method!='z_score')

            preprocessing_options = {
                'strip_headers': strip_headers,
                'strip_strings': strip_strings,
                'convert_dates': convert_dates,
                'date_format': date_format, # Added date_format
                'lowercase_text': lowercase_text,
                'remove_duplicates': remove_duplicates,

                'remove_outliers': remove_outliers,
                'outlier_method': outlier_method,
                'outlier_threshold': outlier_threshold
            }

        with tab_viz:
            visualize_data(st.session_state.get('preprocessed_df', df))

        # Apply preprocessing button
        st.markdown("<br>", unsafe_allow_html=True)
        _, col_btn, _ = st.columns([1, 1, 1])
        if col_btn.button("🧹 Apply Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                df_preprocessed, report = clean_data(df, preprocessing_options)
                st.session_state['preprocessed_df'] = df_preprocessed
                st.session_state['preprocessing_report'] = report

            st.success("Preprocessing completed!")

            # Display preprocessing report
            st.write("### Preprocessing Report")
            for item in st.session_state['preprocessing_report']:
                st.write(item)

            # Show before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before:**")
                st.write(f"Shape: {df.shape}")
                st.write(f"Columns: {df.columns.tolist()}")
            with col2:
                st.write("**After:**")
                st.write(f"Shape: {df_preprocessed.shape}")
                st.write(f"Columns: {df_preprocessed.columns.tolist()}")

            st.divider()

        # Use preprocessed data if available, otherwise use original
        if 'preprocessed_df' in st.session_state:
            df = st.session_state['preprocessed_df']
            st.info("ℹ️ Using preprocessed data for null value handling")

        # Display original data
        st.subheader("Original Data")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Show null value statistics
        null_counts = df.isnull().sum()
        null_percent = (df.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0

        if null_counts.sum() > 0:
            st.warning(f"⚠️ Found {null_counts.sum()} null values in the dataset")

            # Create columns for display
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Null Value Summary")
                null_summary = pd.DataFrame({
                    'Column': null_counts.index,
                    'Null Count': null_counts.values,
                    'Percentage': null_percent.values.round(2)
                })
                null_summary = null_summary[null_summary['Null Count'] > 0]
                st.dataframe(null_summary, width="stretch")

            with col2:
                st.write("### Original Data Preview")
                st.dataframe(df.head(10), width="stretch")

            # Filling method selection
            st.subheader("Select Filling Method")

            # identifying columns with nulls
            cols_with_nulls = df.columns[df.isnull().any()].tolist()
            if not cols_with_nulls:
                st.success("🎉 No null values found! Your data is clean.")
            
            else:
                fill_strategy = st.radio(
                    "Null Handling Strategy:",
                    ["Apply Globally (Same method for all)", "Configure Per Column (Individual control)"],
                    horizontal=True
                )

                # Dictionary to store user choices {col_name: method}
                col_fill_choices = {}
                # Dictionary to store custom values {col_name: value}
                col_custom_values = {}

                # Global options list
                fill_options = [
                    "Mean (numeric only)",
                    "Median (numeric only)",
                    "Mode (most frequent)",
                    "Forward Fill",
                    "Backward Fill",
                    "Interpolate",
                    "Zero",
                    "Custom Value",
                    "Drop Column"
                ]

                if fill_strategy == "Apply Globally (Same method for all)":
                    global_method = st.selectbox("Choose how to fill null values:", fill_options)
                    
                    global_custom_val = None
                    if global_method == "Custom Value":
                        global_custom_val = st.text_input("Enter custom value:")

                    # Apply to all null vars
                    for col in cols_with_nulls:
                        col_fill_choices[col] = global_method
                        if global_method == "Custom Value":
                            col_custom_values[col] = global_custom_val

                else:
                    st.write("### 🎛️ Individual Column Configuration")
                    st.info("Select a specific method for each column below.")
                    
                    for col in cols_with_nulls:
                        col_dtype = df[col].dtype
                        # Smart default based on type
                        default_idx = 2 # Mode
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            default_idx = 0 # Mean
                        
                        with st.container():
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                st.markdown(f"**{col}**")
                                st.caption(f"Type: {col_dtype} | Nulls: {df[col].isnull().sum()}")
                            with c2:
                                method = st.selectbox(
                                    f"Method for {col}", 
                                    fill_options, 
                                    key=f"fill_{col}",
                                    index=default_idx
                                )
                                col_fill_choices[col] = method
                                
                                if method == "Custom Value":
                                    val = st.text_input(f"Value for {col}", key=f"custom_{col}")
                                    col_custom_values[col] = val
                            st.divider()

                # Fill button
                if st.button("🚀 Process & Fill Null Values", type="primary"):
                    df_filled = df.copy()
                    
                    try:
                        progress_bar = st.progress(0)
                        processed_count = 0
                        
                        for col, method in col_fill_choices.items():

                                
                            if method == "Drop Column":
                                df_filled.drop(columns=[col], inplace=True)
                                
                            elif method == "Mean (numeric only)":
                                if pd.api.types.is_numeric_dtype(df_filled[col]):
                                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                                
                            elif method == "Median (numeric only)":
                                if pd.api.types.is_numeric_dtype(df_filled[col]):
                                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                                    
                            elif method == "Mode (most frequent)":
                                mode_val = df_filled[col].mode()
                                if len(mode_val) > 0:
                                    df_filled[col].fillna(mode_val[0], inplace=True)
                                    
                            elif method == "Forward Fill":
                                df_filled[col] = df_filled[col].fillna(method='ffill')
                                # Cleanup edge case
                                df_filled[col] = df_filled[col].fillna(method='bfill')
                                
                            elif method == "Backward Fill":
                                df_filled[col] = df_filled[col].fillna(method='bfill')
                                df_filled[col] = df_filled[col].fillna(method='ffill')
                                
                            elif method == "Interpolate":
                                if pd.api.types.is_numeric_dtype(df_filled[col]):
                                    df_filled[col] = df_filled[col].interpolate(method='linear')
                                else:
                                    # Fallback for text
                                    df_filled[col] = df_filled[col].fillna(method='ffill')
                                    
                            elif method == "Zero":
                                df_filled[col] = df_filled[col].fillna(0)
                                
                            elif method == "Custom Value":
                                val = col_custom_values.get(col)
                                if val is not None:
                                    df_filled[col] = df_filled[col].fillna(val)

                            processed_count += 1
                            progress_bar.progress(processed_count / len(col_fill_choices))

                        
                        # Display filled data
                        st.success("✅ Null values processed successfully!")

                        st.subheader("Filled Data")
                        st.write(f"Shape: {df_filled.shape[0]} rows × {df_filled.shape[1]} columns")
                        remaining_nulls = df_filled.isnull().sum().sum()
                        if remaining_nulls > 0:
                            st.info(f"Note: {remaining_nulls} null values remaining (likely due to incompatible methods for the data type).")

                        # Show side-by-side comparison
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("### Before (Original)")
                            st.dataframe(df.head(10), width="stretch")

                        with col2:
                            st.write("### After (Filled)")
                            st.dataframe(df_filled.head(10), width="stretch")

                        # Download button
                        st.subheader("Download Filled Data")

                        # Convert to CSV
                        csv_buffer = BytesIO()
                        df_filled.to_csv(csv_buffer, index=False)
                        
                        # Renaming Feature
                        csv_name = st.text_input("Filename", value="cleaned_data.csv")
                        if not csv_name.lower().endswith(".csv"):
                            csv_name += ".csv"
                        
                        st.download_button(
                            label="📥 Download Clean CSV",
                            data=csv_buffer.getvalue(),
                            file_name=csv_name,
                            mime="text/csv",
                            type="primary"
                        )
                        
                    except Exception as e:
                        st.error(f"Error filling nulls: {str(e)}")
                        import traceback
                        st.markdown(f"```\n{traceback.format_exc()}\n```")

    else:
        st.info("👆 Please upload a CSV file to get started")

        # Display comprehensive workflow instructions
        st.markdown("""
        ### 📋 How to Use CSV Cleaner
        
        Follow these steps to clean and process your CSV data:
        
        #### **Step 1: Upload CSV File**
        - Click on the file uploader above
        - Select a CSV file from your local system
        - The file will be automatically loaded and analyzed
        
        #### **Step 2: Configure Preprocessing Options**
        Select the preprocessing operations you want to apply:
        
        **Text Processing:**
        - ✅ **Strip whitespace from headers** - Removes leading/trailing spaces from column names
        - ✅ **Strip whitespace from text** - Cleans text data in all columns
        - ✅ **Convert date columns to datetime** - Automatically detects and converts date columns
        
        **Data Quality:**
        - ☐ **Lowercase all text entries** - Converts all text to lowercase (optional)
        - ✅ **Remove duplicate rows** - Eliminates identical rows
        
        **Data Optimization:**
        - ✅ **Remove Statistical Outliers** - Removes rows with values far from the mean (Z-Score/IQR)        
        #### **Step 3: Apply Preprocessing**
        - Click the **"🧹 Apply Preprocessing"** button
        - Review the preprocessing report showing what changes were made
        - Check the before/after comparison of data shape and columns
        
        #### **Step 4: Handle Null Values**
        If null values are detected, choose a filling method:
        
        **Available Methods:**
        - **Mean** - Fill numeric columns with mean, non-numeric with mode
        - **Median** - Fill numeric columns with median, non-numeric with mode
        - **Mode** - Fill all columns with most frequent value
        - **Forward Fill** - Propagate last valid value forward
        - **Backward Fill** - Propagate next valid value backward
        - **Interpolate** - Linear interpolation for numeric columns
        - **Zero** - Fill all nulls with 0
        - **Custom Value** - Fill with a user-specified value
        - **Drop Column** - Remove the column entirely
        
        #### **Step 5: Fill Null Values**
        - Select your preferred filling method
        - If using "Custom Value", enter the value in the text input
        - Click **"Fill Null Values"** button
        - Review the before/after comparison
        
        #### **Step 6: Download Cleaned Data**
        - Click the **"Download CSV"** button
        - The cleaned file will be saved as `filled_data.csv`
        
        ---
        
        ### 💡 Tips for Best Results
        - Always apply preprocessing before handling null values
        - Use Mean/Median for numerical datasets
        - Use Mode for categorical data
        - Forward/Backward fill works well for time-series data
        - Review the null value summary to understand data quality issues
        
        ### 🎯 Common Use Cases
        - **Data Science**: Prepare datasets for ML/AI model training
        - **Data Migration**: Standardize data formats before database import
        - **Quality Assurance**: Validate and clean production data
        - **Research**: Normalize research datasets for analysis
        """)

# Run render when executed directly
if __name__ == "__main__":
    render()
