"""
Page 10: Result Summary
Expanded with visual comparisons and clinical implications
"""

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from page_modules.styles import apply_matplotlib_style, COLORS, get_ecg_colors


def render():
    st.title("Results Summary")
    st.markdown("---")

    # ==========================================================================
    # KEY FINDINGS CALLOUT
    # ==========================================================================
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['success']} 0%, #1B4332 100%); 
                    padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;
                    box-shadow: 0 4px 20px rgba(45, 106, 79, 0.3);">
            <h2 style="margin: 0 0 1rem 0; color: white;">🎯 Mission Accomplished</h2>
            <p style="font-size: 1.1rem; opacity: 0.95; margin: 0;">
                Deep Learning models <strong>significantly outperformed</strong> both baseline models 
                and the 2018 benchmark study by Kachuee et al.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================================
    # KEY METRICS
    # ==========================================================================
    
    st.header("📊 Performance Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="MIT-BIH Accuracy",
            value="98.51%",
            delta="+5.11% vs benchmark",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="PTB Accuracy", 
            value="98.42%",
            delta="+2.52% vs benchmark",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="MIT-BIH F1-Score",
            value="0.9236",
            delta="+0.08 vs baseline",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Best Model",
            value="CNN8",
            delta="Transfer Learning",
            delta_color="off"
        )

    st.markdown("---")

    # ==========================================================================
    # MODEL COMPARISON TABLE
    # ==========================================================================
    
    st.header("📋 Full Model Comparison")

    # Custom CSS for centered dataframe
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] table th,
        [data-testid="stDataFrame"] table td {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load results CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "..", "images", "page_10", "results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
        
        # Highlight best results
        def highlight_best(row):
            styles = [''] * len(row)
            if 'CNN8' in str(row.get('Model', '')):
                styles = [f'background-color: rgba(45, 106, 79, 0.2)'] * len(row)
            return styles
        
        styled_df = df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        # Create manual comparison table
        comparison_data = {
            "Model": ["XGBoost (Baseline)", "CNN8 (Deep Learning)", "Benchmark [3]"],
            "MIT-BIH Accuracy": ["97.2%", "98.51%", "93.4%"],
            "MIT-BIH F1": ["0.84", "0.92", "—"],
            "PTB Accuracy": ["97.8%", "98.42%", "95.9%"],
            "PTB F1": ["0.97", "0.98", "—"],
        }
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==========================================================================
    # VISUAL COMPARISON CHART
    # ==========================================================================
    
    st.header("📈 Performance Visualization")
    
    apply_matplotlib_style()
    
    tab_accuracy, tab_f1 = st.tabs(["Accuracy Comparison", "F1-Score by Class"])
    
    with tab_accuracy:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        models = ['Benchmark\n(Kachuee 2018)', 'XGBoost\n(Baseline)', 'CNN8\n(Deep Learning)']
        mit_scores = [93.4, 97.2, 98.51]
        ptb_scores = [95.9, 97.8, 98.42]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mit_scores, width, label='MIT-BIH', color=COLORS['clinical_blue'])
        bars2 = ax.bar(x + width/2, ptb_scores, width, label='PTB', color=COLORS['heart_red'])
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(90, 100)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add horizontal line for benchmark
        ax.axhline(y=93.4, color=COLORS['text_muted'], linestyle='--', alpha=0.5, label='MIT Benchmark')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab_f1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MIT-BIH per-class F1
        mit_classes = ['Class 0\n(Normal)', 'Class 1\n(Supra.)', 'Class 2\n(Ventr.)', 
                       'Class 3\n(Fusion)', 'Class 4\n(Unknown)']
        mit_f1_scores = [0.9924, 0.8606, 0.9600, 0.8171, 0.9876]
        
        ecg_colors = get_ecg_colors()
        colors = [ecg_colors['class_0'], ecg_colors['class_1'], ecg_colors['class_2'],
                  ecg_colors['class_3'], ecg_colors['class_4']]
        
        bars = axes[0].bar(mit_classes, mit_f1_scores, color=colors, edgecolor='white', linewidth=1.5)
        axes[0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[0].set_title('MIT-BIH: Per-Class F1-Score (CNN8)', fontsize=13, fontweight='bold', pad=15)
        axes[0].set_ylim(0.75, 1.05)
        axes[0].axhline(y=0.9236, color=COLORS['heart_red'], linestyle='--', linewidth=2, label='Macro Avg')
        axes[0].legend(loc='lower right')
        
        for bar, score in zip(bars, mit_f1_scores):
            axes[0].annotate(f'{score:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, score),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # PTB per-class F1
        ptb_classes = ['Class 0\n(Normal)', 'Class 1\n(MI)']
        ptb_f1_scores = [0.97, 0.99]
        ptb_colors = [ecg_colors['normal'], ecg_colors['abnormal']]
        
        bars = axes[1].bar(ptb_classes, ptb_f1_scores, color=ptb_colors, edgecolor='white', linewidth=1.5)
        axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1].set_title('PTB: Per-Class F1-Score (CNN8 Transfer)', fontsize=13, fontweight='bold', pad=15)
        axes[1].set_ylim(0.9, 1.05)
        
        for bar, score in zip(bars, ptb_f1_scores):
            axes[1].annotate(f'{score:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, score),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ==========================================================================
    # KEY FINDINGS
    # ==========================================================================
    
    st.header("🔬 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                        border-left: 4px solid {COLORS['success']}; height: 100%;">
                <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">✅ Strengths</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Benchmark exceeded</strong> on both datasets</li>
                    <li><strong>Transfer learning</strong> effective for PTB with limited data</li>
                    <li><strong>High recall</strong> for critical classes (Class 0, Class 4)</li>
                    <li><strong>Interpretable</strong> — SHAP confirms focus on R-peaks</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.5rem; border-radius: 12px; 
                        border-left: 4px solid {COLORS['warning']}; height: 100%;">
                <h4 style="color: {COLORS['clinical_blue']}; margin-top: 0;">⚠️ Limitations</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Class 1 & 3</strong> have lower F1 due to imbalance</li>
                    <li><strong>False negatives</strong> present for abnormal classes</li>
                    <li><strong>Data quality</strong> concerns in extreme RR-distance samples</li>
                    <li><strong>Single-lead ECG</strong> limits generalization</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ==========================================================================
    # CLINICAL IMPLICATIONS
    # ==========================================================================
    
    st.header("🏥 Clinical Implications")
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 2rem; border-radius: 16px; color: white;">
            <h3 style="margin-top: 0; color: white;">Potential Clinical Workflow Integration</h3>
            <p style="font-size: 1.05rem; opacity: 0.95;">
                While this is a research prototype, the approach could support an 
                <strong>ECG triage workflow</strong>:
            </p>
            <ol style="font-size: 1rem;">
                <li><strong>Automated Screening</strong> — Model performs initial screening of incoming ECG signals</li>
                <li><strong>Flagging System</strong> — "Abnormal" or high-risk patterns are flagged for review</li>
                <li><strong>Clinical Review</strong> — Cardiologists confirm or dismiss the model's suggestions</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                        text-align: center; border-top: 3px solid {COLORS['clinical_blue_light']};">
                <span style="font-size: 2rem;">⚡</span>
                <h4 style="margin: 0.5rem 0;">Reduced Workload</h4>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                    Automates routine screening, letting clinicians focus on complex cases
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                        text-align: center; border-top: 3px solid {COLORS['heart_red']};">
                <span style="font-size: 2rem;">🔍</span>
                <h4 style="margin: 0.5rem 0;">Early Detection</h4>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                    Identifies subtle abnormalities that might be missed in manual review
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background: {COLORS['card_bg']}; padding: 1.25rem; border-radius: 10px; 
                        text-align: center; border-top: 3px solid {COLORS['success']};">
                <span style="font-size: 2rem;">⏱️</span>
                <h4 style="margin: 0.5rem 0;">Faster Decisions</h4>
                <p style="font-size: 0.9rem; color: {COLORS['text_secondary']}; margin: 0;">
                    Accelerates clinical decision-making in time-critical situations
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Important disclaimer
    st.warning(
        """
        **⚕️ Important Note:** This model is designed as a **decision-support tool**, not a replacement 
        for qualified medical professionals. Expert validation remains essential for patient safety 
        and handling edge cases.
        """
    )
