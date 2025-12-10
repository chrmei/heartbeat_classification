"""
Shared styling module for Heartbeat Classification Streamlit App.
Provides consistent CSS injection, color palettes, and styling utilities.
"""

import streamlit as st

# =============================================================================
# COLOR PALETTE - Medical/Scientific Theme
# =============================================================================

COLORS = {
    # Primary colors
    "heart_red": "#E63946",
    "heart_red_light": "#F8D7DA",
    "heart_red_dark": "#C1121F",
    
    # Clinical blues
    "clinical_blue": "#1D3557",
    "clinical_blue_light": "#457B9D",
    "clinical_blue_lighter": "#A8DADC",
    
    # Neutral tones
    "background": "#FAFAFA",
    "card_bg": "#F1F3F5",
    "border": "#DEE2E6",
    "text_primary": "#1D3557",
    "text_secondary": "#495057",
    "text_muted": "#6C757D",
    
    # Status colors
    "success": "#2D6A4F",
    "success_light": "#D8F3DC",
    "warning": "#E9C46A",
    "warning_light": "#FFF3CD",
    "error": "#E63946",
    "error_light": "#F8D7DA",
    
    # ECG Signal colors
    "ecg_normal": "#2D6A4F",
    "ecg_abnormal": "#E63946",
    "ecg_grid": "#E9ECEF",
}

# =============================================================================
# CSS INJECTION FUNCTIONS
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for the entire app. Call once in main app.py."""
    st.markdown(get_base_css(), unsafe_allow_html=True)


def get_base_css():
    """Return the base CSS styling for the app."""
    return f"""
    <style>
    /* =================================================================
       GLOBAL STYLES
       ================================================================= */
    
    /* Improve overall typography */
    .stMarkdown {{
        line-height: 1.7;
    }}
    
    /* Header styling */
    h1 {{
        color: {COLORS['clinical_blue']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {COLORS['heart_red']};
        margin-bottom: 1.5rem !important;
    }}
    
    h2 {{
        color: {COLORS['clinical_blue']} !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }}
    
    h3 {{
        color: {COLORS['clinical_blue_light']} !important;
        font-weight: 600 !important;
    }}
    
    /* =================================================================
       SIDEBAR STYLES
       ================================================================= */
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['clinical_blue']} 0%, #264653 100%);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: white !important;
    }}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white !important;
        border-bottom: none;
    }}
    
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.2);
    }}
    
    [data-testid="stSidebar"] label {{
        color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Radio button styling in sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        color: rgba(255, 255, 255, 0.85) !important;
        transition: all 0.2s ease;
    }}
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        color: white !important;
        padding-left: 5px;
    }}
    
    /* =================================================================
       METRIC CARDS
       ================================================================= */
    
    [data-testid="stMetric"] {{
        background-color: {COLORS['card_bg']};
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['heart_red']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    [data-testid="stMetric"] label {{
        color: {COLORS['text_secondary']} !important;
        font-weight: 500;
    }}
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS['clinical_blue']} !important;
        font-weight: 700;
    }}
    
    /* =================================================================
       EXPANDERS
       ================================================================= */
    
    .streamlit-expanderHeader {{
        font-weight: 600;
        color: {COLORS['clinical_blue']} !important;
        background-color: {COLORS['card_bg']};
        border-radius: 8px;
    }}
    
    .streamlit-expanderContent {{
        border-left: 3px solid {COLORS['clinical_blue_lighter']};
        padding-left: 1rem;
    }}
    
    /* =================================================================
       TABS
       ================================================================= */
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%);
        padding: 0.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(29, 53, 87, 0.3);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 600;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8) !important;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['heart_red']} !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(230, 57, 70, 0.4);
    }}
    
    /* =================================================================
       DATAFRAMES & TABLES
       ================================================================= */
    
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    [data-testid="stDataFrame"] table th {{
        background-color: {COLORS['clinical_blue']} !important;
        color: white !important;
        font-weight: 600;
        text-align: center !important;
    }}
    
    [data-testid="stDataFrame"] table td {{
        text-align: center !important;
    }}
    
    /* =================================================================
       BUTTONS
       ================================================================= */
    
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: none;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.3);
    }}
    
    /* =================================================================
       ALERTS & CALLOUTS
       ================================================================= */
    
    .stAlert {{
        border-radius: 10px;
        border-left-width: 4px;
    }}
    
    /* Success styling */
    [data-testid="stAlert"][data-baseweb="notification"] {{
        border-radius: 10px;
    }}
    
    /* =================================================================
       IMAGES
       ================================================================= */
    
    [data-testid="stImage"] {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    /* =================================================================
       DIVIDERS
       ================================================================= */
    
    hr {{
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, {COLORS['heart_red']} 0%, {COLORS['clinical_blue_lighter']} 50%, transparent 100%);
    }}
    
    /* =================================================================
       HEARTBEAT ANIMATION
       ================================================================= */
    
    @keyframes heartbeat {{
        0% {{ transform: scale(1); }}
        14% {{ transform: scale(1.1); }}
        28% {{ transform: scale(1); }}
        42% {{ transform: scale(1.1); }}
        70% {{ transform: scale(1); }}
    }}
    
    .heartbeat-icon {{
        display: inline-block;
        animation: heartbeat 1.5s ease-in-out infinite;
        color: {COLORS['heart_red']};
    }}
    
    /* =================================================================
       CUSTOM CARD COMPONENT
       ================================================================= */
    
    .custom-card {{
        background-color: {COLORS['card_bg']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid {COLORS['border']};
        margin-bottom: 1rem;
    }}
    
    .custom-card-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLORS['clinical_blue']};
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['clinical_blue_lighter']};
    }}
    
    /* =================================================================
       HERO SECTION
       ================================================================= */
    
    .hero-container {{
        background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%);
        border-radius: 16px;
        padding: 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(29, 53, 87, 0.3);
    }}
    
    .hero-title {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
        border: none !important;
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }}
    
    .hero-metric {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }}
    
    .hero-metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['heart_red_light']};
    }}
    
    .hero-metric-label {{
        font-size: 0.9rem;
        opacity: 0.85;
    }}
    
    /* =================================================================
       NAVIGATION PROGRESS
       ================================================================= */
    
    .nav-section {{
        margin-bottom: 0.5rem;
    }}
    
    .nav-section-header {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255, 255, 255, 0.6);
        margin-bottom: 0.25rem;
        padding-left: 0.5rem;
    }}
    
    .nav-progress {{
        height: 4px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 2px;
        margin: 1rem 0;
        overflow: hidden;
    }}
    
    .nav-progress-bar {{
        height: 100%;
        background: {COLORS['heart_red']};
        border-radius: 2px;
        transition: width 0.3s ease;
    }}
    
    /* =================================================================
       CITATION STYLING
       ================================================================= */
    
    .citation {{
        font-size: 0.85rem;
        color: {COLORS['text_muted']};
        background-color: {COLORS['card_bg']};
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid {COLORS['clinical_blue_lighter']};
    }}
    
    .citation a {{
        color: {COLORS['clinical_blue_light']};
    }}
    
    </style>
    """


# =============================================================================
# COMPONENT HELPER FUNCTIONS
# =============================================================================

def render_hero_section(title: str, subtitle: str, metrics: list[dict] = None):
    """
    Render a hero section with title, subtitle, and optional metrics.
    
    Args:
        title: Main title text
        subtitle: Subtitle/description text
        metrics: List of dicts with 'value' and 'label' keys
    """
    metrics_html = ""
    if metrics:
        metrics_items = "".join([
            f'<div class="hero-metric"><div class="hero-metric-value">{m["value"]}</div><div class="hero-metric-label">{m["label"]}</div></div>'
            for m in metrics
        ])
        metrics_html = f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1.5rem;">{metrics_items}</div>'
    
    html = f'<div class="hero-container"><div class="hero-title"><span class="heartbeat-icon">❤️</span> {title}</div><div class="hero-subtitle">{subtitle}</div>{metrics_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_card(title: str, content: str, icon: str = None):
    """
    Render a styled card component.
    
    Args:
        title: Card header title
        content: Card body content (can be HTML)
        icon: Optional emoji icon for the header
    """
    icon_html = f"{icon} " if icon else ""
    html = f"""
    <div class="custom-card">
        <div class="custom-card-header">{icon_html}{title}</div>
        <div>{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(metrics: list[dict]):
    """
    Render a row of metrics in a grid layout.
    
    Args:
        metrics: List of dicts with 'value', 'label', and optional 'delta' keys
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            delta = metric.get('delta')
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=delta
            )


def render_citations(citations: list[dict]):
    """
    Render a formatted citations section.
    
    Args:
        citations: List of dicts with 'id', 'text', and optional 'url' keys
    """
    citation_items = []
    for c in citations:
        url_html = f' <a href="{c["url"]}" target="_blank">🔗</a>' if c.get('url') else ''
        citation_items.append(f"[{c['id']}] {c['text']}{url_html}")
    
    html = f"""
    <div class="citation">
        {'<br><br>'.join(citation_items)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_nav_progress(current_page: int, total_pages: int):
    """
    Render a navigation progress bar.
    
    Args:
        current_page: Current page number (1-indexed)
        total_pages: Total number of pages
    """
    progress_pct = (current_page / total_pages) * 100
    html = f"""
    <div class="nav-progress">
        <div class="nav-progress-bar" style="width: {progress_pct}%;"></div>
    </div>
    <div style="text-align: center; font-size: 0.8rem; color: rgba(255,255,255,0.7);">
        Page {current_page} of {total_pages}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(text: str):
    """Render a navigation section header."""
    html = f'<div class="nav-section-header">{text}</div>'
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MATPLOTLIB THEME CONFIGURATION
# =============================================================================

def get_matplotlib_style():
    """
    Return a dictionary of matplotlib rcParams for consistent styling.
    Apply with: plt.rcParams.update(get_matplotlib_style())
    """
    return {
        # Figure
        'figure.facecolor': COLORS['background'],
        'figure.edgecolor': COLORS['background'],
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        
        # Axes
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['border'],
        'axes.labelcolor': COLORS['text_primary'],
        'axes.titlecolor': COLORS['clinical_blue'],
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'medium',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid
        'grid.color': COLORS['ecg_grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Ticks
        'xtick.color': COLORS['text_secondary'],
        'ytick.color': COLORS['text_secondary'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        
        # Legend
        'legend.frameon': True,
        'legend.facecolor': COLORS['card_bg'],
        'legend.edgecolor': COLORS['border'],
        'legend.fontsize': 10,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Font
        'font.family': 'sans-serif',
        'font.size': 11,
    }


def get_ecg_colors():
    """Return color palette for ECG visualizations."""
    return {
        'normal': COLORS['ecg_normal'],
        'abnormal': COLORS['ecg_abnormal'],
        'class_0': '#2D6A4F',  # Normal - Green
        'class_1': '#E9C46A',  # Supraventricular - Yellow
        'class_2': '#E76F51',  # Ventricular - Orange
        'class_3': '#9B2226',  # Fusion - Dark Red
        'class_4': '#457B9D',  # Unknown - Blue
        'grid': COLORS['ecg_grid'],
        'prediction_correct': '#2D6A4F',
        'prediction_wrong': '#E63946',
    }


def apply_matplotlib_style():
    """Apply the custom matplotlib style globally."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(get_matplotlib_style())

