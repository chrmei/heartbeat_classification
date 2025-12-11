"""
Page 2: Presentation of the data (volume, architecture, etc.)
Data analysis using DataVisualization figures
Description and justification of the pre-processing carried out
Refactored with proper caching and loading
"""

import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from page_modules.styles import apply_matplotlib_style, COLORS, get_ecg_colors

# Base paths
APP_DIR = Path(__file__).parent.parent
IMAGES_DIR = APP_DIR / "images" / "page_2"
DATA_DIR = APP_DIR.parent / "data" / "original"


# =============================================================================
# IMAGE HELPER FUNCTIONS
# =============================================================================

def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image."""
    ext = image_path.suffix.lower()
    mime_types = {".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime = mime_types.get(ext, "image/png")
    b64 = get_image_base64(image_path)
    
    caption_html = f'<p style="text-align: center; font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">{caption}</p>' if caption else ''
    
    return f'''
        <img src="data:{mime};base64,{b64}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 8px;">
        {caption_html}
    '''


# =============================================================================
# CACHED DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_mitbih_data():
    """Load and combine MIT-BIH train and test datasets with caching."""
    df_train = pd.read_csv(DATA_DIR / "mitbih_train.csv", header=None)
    df_test = pd.read_csv(DATA_DIR / "mitbih_test.csv", header=None)
    
    # Combine datasets
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    return df_combined


@st.cache_data
def load_ptbdb_data():
    """Load and combine PTB normal and abnormal datasets with caching."""
    df_normal = pd.read_csv(DATA_DIR / "ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv(DATA_DIR / "ptbdb_abnormal.csv", header=None)
    
    # Combine datasets
    df_combined = pd.concat([df_normal, df_abnormal], axis=0, ignore_index=True)
    
    return df_combined, df_normal


@st.cache_data
def get_class_dataframes(df, class_column=187):
    """Split dataframe by class labels."""
    classes = {}
    unique_classes = sorted(df[class_column].unique())
    
    for cls in unique_classes:
        df_cls = df.loc[df[class_column] == cls]
        df_cls_plot = df_cls.drop(class_column, axis=1)
        classes[int(cls)] = df_cls_plot
    
    return classes


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_ecg_examples(df_class, class_label, dataset_name, n_examples=3, seed=42):
    """Plot ECG examples for a given class with consistent styling."""
    apply_matplotlib_style()
    ecg_colors = get_ecg_colors()
    
    examples = df_class.sample(n=min(n_examples, len(df_class)), random_state=seed)
    time_points = df_class.columns
    
    cols = st.columns(n_examples)
    
    # Select color based on class
    if dataset_name == "MIT":
        color = ecg_colors.get(f'class_{class_label}', COLORS['clinical_blue'])
    else:
        color = ecg_colors['normal'] if class_label == 0 else ecg_colors['abnormal']
    
    for i in range(min(n_examples, len(examples))):
        row = examples.iloc[i]
        
        with cols[i]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_points, row.values, color=color, linewidth=1.2)
            ax.fill_between(time_points, row.values, alpha=0.1, color=color)
            
            ax.set_title(f"Example {i+1}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time Point", fontsize=10)
            ax.set_ylabel("Amplitude", fontsize=10)
            ax.set_xlim(0, len(time_points))
            
            # ECG-style grid
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
            ax.minorticks_on()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render():
    # Hero-style header for dark mode readability (using CSS classes from styles.py)
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">📊 Data Overview</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Load data with spinner
    with st.spinner("Loading datasets..."):
        df_mitbih = load_mitbih_data()
        df_ptbdb, df_ptbdb_normal = load_ptbdb_data()
        
        mit_classes = get_class_dataframes(df_mitbih)
        ptb_classes = get_class_dataframes(df_ptbdb)

    # ==========================================================================
    # DATASET INFORMATION SECTION
    # ==========================================================================
    
    # Hero-style container with title and metrics inside (like page 1 "Heartbeat Classification")
    st.markdown(
        f'''
        <div class="hero-container">
            <div class="hero-title" style="font-size: 1.8rem;">📋 Dataset Information</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                <div class="hero-metric">
                    <div class="hero-metric-value">{len(df_mitbih):,}</div>
                    <div class="hero-metric-label">MIT-BIH Samples</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">{len(df_ptbdb):,}</div>
                    <div class="hero-metric-label">PTB Samples</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">187</div>
                    <div class="hero-metric-label">Features</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">2</div>
                    <div class="hero-metric-label">Classification Tasks</div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    with st.expander("📄 Dataset Details", expanded=False):
        st.markdown(
            """
            - **Problem Type:** Supervised Classification (labeled data)  
            - **Input Data:** Preprocessed ECG signals — each sample represents one heartbeat centered around the R-peak  
            - **Tasks:**   
                - **MIT-BIH Arrhythmia:** 5-class classification  
                - **PTB Diagnostic ECG:** Binary classification (MI vs. healthy)  
            - **Data Structure:** Each sample contains **188 columns**:   
                - **Columns 0-186:** 187 time-series points (~1.2 heartbeat cycles)  
                - **Column 187:** Target label (Class ID) 
            """
        )

    with st.expander("📋 Preprocessing Pipeline (Kachuee et al.)", expanded=False):
        st.markdown(
            """
            The datasets [4] we received had already been processed following the pipeline defined by Kachuee et al. [3]:
            
            1. **Windowing** — Splitting continuous ECG signals into 10-second segments  
            2. **Normalization** — Scaling signal amplitudes to the range (0, 1)  
            3. **R-Peak Detection** — Identifying local maxima corresponding to R-peaks  
            4. **Cropping** — Extracting a window around each peak to capture the full P-Q-R-S-T complex  
            5. **Padding** — Applying zero-padding to standardize each heartbeat to **187 time-steps**
            """
        )

    # Data preview tabs
    tab_mit, tab_ptb = st.tabs(["MIT-BIH Preview", "PTB Preview"])
    
    with tab_mit:
        st.dataframe(df_mitbih.head(5), width='stretch')
    
    with tab_ptb:
        st.dataframe(df_ptbdb_normal.head(5), width='stretch')

    st.divider()

    # ==========================================================================
    # MIT-BIH DATASET SECTION
    # ==========================================================================
    
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">🫀 MIT-BIH Arrhythmia Dataset</div>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("View MIT-BIH Dataset Details", expanded=False):
        col_info, col_dist = st.columns([1, 1])
        
        with col_info:
            st.markdown(
                """
                **Source:** ECG recordings from 47 subjects
                
                **Classes (5 categories):**
                | Class | Label | Description |
                |-------|-------|-------------|
                | 0 | N | Normal beat |
                | 1 | S | Supraventricular/Atrial premature |
                | 2 | V | Premature ventricular contraction |
                | 3 | F | Fusion of ventricular and normal |
                | 4 | Q | Unclassifiable/fusion of paced and normal |
                
                **Dataset Properties:**
                - **109,446** heartbeat samples
                - Numerical, normalized, preprocessed
                - No missing values or duplicates
                """
            )
            
            st.warning(
                "**⚠️ Key Challenge: Severe Class Imbalance**\n\n"
                "Data augmentation (SMOTE) is necessary to prevent model bias toward majority class."
            )
        
        with col_dist:
            mit_dist_img = get_image_html(IMAGES_DIR / "MIT_combined.png", "MIT-BIH Class Distribution", "MIT-BIH Class Distribution")
            st.markdown(
                f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{mit_dist_img}</div>',
                unsafe_allow_html=True
            )

        # ECG Examples
        st.subheader("📊 MIT-BIH — ECG Signal Examples")
        
        mit_class_names = {
            0: "Normal (N)",
            1: "Supraventricular (S)", 
            2: "Ventricular (V)",
            3: "Fusion (F)",
            4: "Unknown (Q)"
        }
        
        selected_mit_classes = st.multiselect(
            "Select classes to visualize:",
            options=list(mit_class_names.keys()),
            default=[0],
            format_func=lambda x: f"Class {x}: {mit_class_names[x]}",
            key="mit_class_select"
        )

        for cls in selected_mit_classes:
            st.markdown(f"#### Class {cls}: {mit_class_names[cls]}")
            if cls in mit_classes:
                plot_ecg_examples(mit_classes[cls], cls, "MIT")
            else:
                st.warning(f"No data available for class {cls}")

    st.divider()

    # ==========================================================================
    # PTB DATASET SECTION
    # ==========================================================================
    
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">❤️‍🩹 PTB Diagnostic ECG Dataset</div>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("View PTB Dataset Details", expanded=False):
        col_info2, col_dist2 = st.columns([1, 1])
        
        with col_info2:
            st.markdown(
                """
                **Source:** ECG recordings from 290 subjects
                - 148 Myocardial Infarction (MI) patients
                - 53 healthy controls
                - 90 patients with other cardiac diseases
                
                **Classes (Binary):**
                | Class | Label | Description |
                |-------|-------|-------------|
                | 0 | Normal | Healthy heartbeat |
                | 1 | Abnormal | Myocardial Infarction (MI) |
                
                **Dataset Properties:**
                - **14,552** samples total
                - After removing 7 duplicates → **14,545** samples
                - Numerical, normalized, preprocessed
                - No missing values
                """
            )
            
            st.warning(
                "**⚠️ Key Challenge: Imbalanced Classes**\n\n"
                "The \"Normal\" class is the minority — requires careful handling during training."
            )
        
        with col_dist2:
            ptb_dist_img = get_image_html(IMAGES_DIR / "PTB_combined.png", "PTB Class Distribution", "PTB Class Distribution")
            st.markdown(
                f'<div style="min-width: 200px; max-width: 400px; text-align: center;">{ptb_dist_img}</div>',
                unsafe_allow_html=True
            )

        # ECG Examples
        st.subheader("📊 PTB — ECG Signal Examples")
        
        ptb_class_names = {
            0: "Normal (Healthy)",
            1: "Abnormal (MI)"
        }
        
        selected_ptb_classes = st.multiselect(
            "Select classes to visualize:",
            options=list(ptb_class_names.keys()),
            default=[0],
            format_func=lambda x: f"Class {x}: {ptb_class_names[x]}",
            key="ptb_class_select"
        )

        for cls in selected_ptb_classes:
            st.markdown(f"#### Class {cls}: {ptb_class_names[cls]}")
            if cls in ptb_classes:
                plot_ecg_examples(ptb_classes[cls], cls, "PTB")
            else:
                st.warning(f"No data available for class {cls}")

    # Citations at the end of the page
    st.markdown("---")
    with st.expander("📚 Citations", expanded=False):
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {COLORS['clinical_blue_lighter']};">
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[3]</strong> Kachuee M, Fazeli S, Sarrafzadeh M. (2018). <em>ECG Heartbeat Classification: 
                    A Deep Transferable Representation</em>. 
                    <a href="https://arxiv.org/abs/1805.00794" style="color: {COLORS['clinical_blue_light']};">arXiv:1805.00794</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0;">
                    <strong>[4]</strong> PhysioNet. MIT-BIH Arrhythmia Database & PTB Diagnostic ECG Database.
                    <a href="https://physionet.org/" style="color: {COLORS['clinical_blue_light']};">https://physionet.org/</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
