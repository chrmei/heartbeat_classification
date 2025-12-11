"""
Page 7: DL intro
Description of Deep Learning approach and models used
"""

import streamlit as st
import base64
from pathlib import Path
from page_modules.styles import COLORS

# Base path for images
IMAGES_DIR = Path(__file__).parent.parent / "images" / "page_7"


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_image_html(image_path: Path, alt: str = "", caption: str = "") -> str:
    """Generate HTML img tag with base64 encoded image."""
    ext = image_path.suffix.lower()
    mime_types = {
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    mime = mime_types.get(ext, "image/png")
    b64 = get_image_base64(image_path)

    caption_html = (
        f'<p style="text-align: center; font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;">{caption}</p>'
        if caption
        else ""
    )

    return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="max-width: 100%; height: auto; border-radius: 8px;">{caption_html}'


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">🧠 Deep Learning Models</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # =============================================================================
    # INTRODUCTION SECTION
    # =============================================================================
    st.markdown(
        '<div class="hero-container" style="padding: 1.5rem;">'
        '<div class="hero-title" style="font-size: 1.8rem;">📋 Introduction</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Description in styled container
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white;
                    box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3); margin-bottom: 1.5rem;">
            <p style="margin: 0; opacity: 0.95;">
                Deep Learning models offer powerful capabilities for ECG heartbeat classification. 
                This section explores different DL architectures, inspired by published research, 
                and our optimization strategies to achieve state-of-the-art performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =============================================================================
    # WORKFLOW SECTION
    # =============================================================================
    with st.expander("📊 View Workflow Details", expanded=False):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">🔄 Deep Learning Workflow</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25); height: auto;">
                    <h4 style="margin: 0 0 0.75rem 0; color: white;">🫀 MIT Dataset</h4>
                    <ul style="opacity: 0.95; margin-bottom: 0; padding-left: 1.25rem;">
                        <li>Test different DL model types (inspiration from lessons, [3] and [6])
                            <ul style="margin-top: 0.25rem;">
                                <li><em>Goal:</em> Find best option for arrhythmia classification</li>
                            </ul>
                        </li>
                        <li style="margin-top: 0.5rem;">Optimization of hyperparameters and training procedure
                            <ul style="margin-top: 0.25rem;">
                                <li><em>Goal:</em> Outperform models presented in publications for arrhythmia classification</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(69, 123, 157, 0.25); height: auto;">
                    <h4 style="margin: 0 0 0.75rem 0; color: white;">❤️ PTB Dataset</h4>
                    <ul style="opacity: 0.95; margin-bottom: 0; padding-left: 1.25rem;">
                        <li>Transfer learning: Use pretrained MIT model (best option found) and retrain for PTB classification task
                            <ul style="margin-top: 0.25rem;">
                                <li><em>Goal:</em> Generate DL model for MI detection and benefit from MIT modeling phase</li>
                            </ul>
                        </li>
                        <li style="margin-top: 0.5rem;">Optimization of hyperparameters and training procedure
                            <ul style="margin-top: 0.25rem;">
                                <li><em>Goal:</em> Outperform models presented in publications for MI detection</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Reference architectures tabs
        tab1, tab2 = st.tabs(["📄 Reference [3]", "📄 Reference [6]"])

        with tab1:
            image_path = IMAGES_DIR / "2018.png"
            if image_path.exists():
                st.markdown(
                    f"""
                    <div style="min-width: 200px; max-width: 400px; text-align: center; margin: 0 auto;">
                        {get_image_html(image_path, "Architecture from 2018 paper", "CNN Architecture from Kachuee et al. (2018) [3]")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ Image not found")

        with tab2:
            image_path = IMAGES_DIR / "2020.png"
            if image_path.exists():
                st.markdown(
                    f"""
                    <div style="min-width: 200px; max-width: 400px; text-align: center; margin: 0 auto;">
                        {get_image_html(image_path, "Architecture from 2020 paper", "DL Architecture from Murat et al. (2020) [6]")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ Image not found")

    # =============================================================================
    # TESTED DL MODELS SECTION
    # =============================================================================
    with st.expander("🔬 View Tested DL Models for MIT Dataset", expanded=False):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">🧪 Tested DL Models for MIT Dataset</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25); height: 150px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: white; font-size: 1rem;">🔗 Dense Neural Network (DNN)</h4>
                    <p style="opacity: 0.9; margin: 0; font-size: 0.9rem;">
                        DNN architecture inspired by lessons [5]
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(69, 123, 157, 0.25); height: 150px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: white; font-size: 1rem;">📊 Convolutional Neural Network (CNN)</h4>
                    <p style="opacity: 0.9; margin: 0; font-size: 0.9rem;">
                        CNN architectures presented in [3, 6] rebuilt
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['success']} 0%, #1B4332 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            box-shadow: 0 4px 15px rgba(45, 106, 79, 0.25); height: 150px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: white; font-size: 1rem;">🔄 Long Short-Term Memory (LSTM)</h4>
                    <p style="opacity: 0.9; margin: 0; font-size: 0.9rem;">
                        LSTM architectures presented in [6] rebuilt
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Architecture images tabs
        tab1, tab2, tab3 = st.tabs(["📄 CNN [3]", "📄 CNN [6]", "📄 LSTM [6]"])

        with tab1:
            image_path = IMAGES_DIR / "cnn_2018.png"
            if image_path.exists():
                st.markdown(
                    f"""
                    <div style="min-width: 200px; max-width: 400px; text-align: center; margin: 0 auto;">
                        {get_image_html(image_path, "CNN Architecture from [3]", "CNN Architecture from Kachuee et al. (2018)")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ CNN [3] image not found")

        with tab2:
            image_path = IMAGES_DIR / "cnn4_2020.png"
            if image_path.exists():
                st.markdown(
                    f"""
                    <div style="min-width: 200px; max-width: 400px; text-align: center; margin: 0 auto;">
                        {get_image_html(image_path, "CNN Architecture from [6]", "CNN Architecture from Murat et al. (2020)")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ CNN [6] image not found")

        with tab3:
            image_path = IMAGES_DIR / "lstm_2020.png"
            if image_path.exists():
                st.markdown(
                    f"""
                    <div style="min-width: 200px; max-width: 400px; text-align: center; margin: 0 auto;">
                        {get_image_html(image_path, "LSTM Architecture from [6]", "LSTM Architecture from Murat et al. (2020)")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("⚠️ LSTM [6] image not found")

    # =============================================================================
    # TRAINING SETUP SECTION
    # =============================================================================
    with st.expander("⚙️ View Training Setup Details", expanded=False):
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {COLORS['clinical_blue_light']} 0%, #1D3557 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; color: white;
                        box-shadow: 0 4px 15px rgba(69, 123, 157, 0.3); margin-bottom: 1rem;">
                <h4 style="color: white; margin: 0;">⚙️ Training Setup</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">📊 SMOTE</h4>
                    <p style="margin: 0; opacity: 0.9;">
                        Oversampling technique for all DL training procedures to handle class imbalance
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">📉 Loss Minimization</h4>
                    <p style="margin: 0; opacity: 0.9;">
                        Training procedure uses minimization of loss function → class sensitive
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                            padding: 1.25rem; border-radius: 12px; color: white;
                            border-left: 4px solid {COLORS['heart_red']};
                            box-shadow: 0 4px 15px rgba(29, 53, 87, 0.25);">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">🛑 Early Stopping</h4>
                    <p style="margin: 0; opacity: 0.9;">
                        Applied when training loss did not further decrease to prevent overfitting
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # =============================================================================
    # CITATIONS SECTION
    # =============================================================================
    with st.expander("📚 Citations", expanded=False):
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid {COLORS['clinical_blue_lighter']};">
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[3]</strong> ECG Heartbeat Classification: A Deep Transferable Representation; 
                    M. Kachuee, S. Fazeli, M. Sarrafzadeh (2018); CoRR; 
                    <a href="https://doi.org/10.48550/arXiv.1805.00794" style="color: {COLORS['clinical_blue_light']};">doi: 10.48550/arXiv.1805.00794</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[4]</strong> <a href="https://www.datasci.com/solutions/cardiovascular/ecg-research" style="color: {COLORS['clinical_blue_light']};">https://www.datasci.com/solutions/cardiovascular/ecg-research</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[5]</strong> Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023; 
                    Y. Ansari, O. Mourad, K. Qaraqe, E. Serpedin (2023); 
                    <a href="https://doi.org/10.3389/fphys.2023.1246746" style="color: {COLORS['clinical_blue_light']};">doi: 10.3389/fphys.2023.1246746</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0.75rem;">
                    <strong>[6]</strong> Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review; 
                    F. Murat, O. Yildirim, M. Talo, U. B. Baloglu, Y. Demir, U. R. Acharya (2020); 
                    Computers in Biology and Medicine; 
                    <a href="https://doi.org/10.1016/j.compbiomed.2020.103726" style="color: {COLORS['clinical_blue_light']};">doi: 10.1016/j.compbiomed.2020.103726</a>
                </p>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin-bottom: 0;">
                    <strong>[7]</strong> ECG-based heartbeat classification for arrhythmia detection: A survey; 
                    E. J. da S. Luz, W. R. Schwartz, G. Cámara-Chávez, D. Menotti (2015); 
                    Computer Methods and Programs in Biomedicine; 
                    <a href="https://doi.org/10.1016/j.cmpb.2015.12.008" style="color: {COLORS['clinical_blue_light']};">doi: 10.1016/j.cmpb.2015.12.008</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render()
