"""
Page 14: Authors
Team members and their social links
"""

import streamlit as st
import base64
from pathlib import Path
from page_modules.styles import COLORS

# Base path for images
IMAGES_DIR = Path(__file__).parent.parent / "images" / "page_14"


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_author_image_html(image_filename: str, author_name: str, size: int = 120) -> str:
    """Generate HTML for round author profile picture, or fallback to emoji."""
    if not image_filename:
        return '<div style="font-size: 4rem; margin-bottom: 1rem;">👤</div>'

    image_path = IMAGES_DIR / image_filename
    b64 = get_image_base64(image_path)

    if b64 is None:
        return '<div style="font-size: 4rem; margin-bottom: 1rem;">👤</div>'

    # Determine mime type from extension
    ext = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_types.get(ext, "image/jpeg")

    # Modern style: subtle shadow, no border - single line to avoid HTML issues
    return f'<img src="data:{mime};base64,{b64}" alt="{author_name}" style="width: {size}px; height: {size}px; border-radius: 50%; object-fit: cover; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(0,0,0,0.12);">'


def render():
    # Hero-style header
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 2rem;">'
        '<div class="hero-title" style="justify-content: center;">👥 Authors</div>'
        '<div class="hero-subtitle">Meet the team behind this project</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ==========================================================================
    # AUTHORS SECTION
    # ==========================================================================

    # Authors data (alphabetical order)
    # Place author images in: app/images/page_14/
    authors = [
        {
            "name": "Christian Meister",
            "image": "0001.jpg",  # Place image in app/images/page_14/
            "linkedin": "https://www.linkedin.com/in/christian--meister/",
            "github": "https://github.com/chrmei",
            "dagshub": "https://dagshub.com/chrmei",
        },
        {
            "name": "Julia Schmidt",
            "image": "0002.jpg",  # Place image in app/images/page_14/
            "linkedin": "https://www.linkedin.com/in/julia-schmidt-554aa7238/",
            "github": "https://github.com/julia-schmidtt",
            "dagshub": "https://dagshub.com/julia-schmidtt",
        },
        {
            "name": "Tzu-Jung Huang",
            "image": "0003.jpg",  # Place image in app/images/page_14/
            "linkedin": "https://www.linkedin.com/in/tzu-jung-huang-kiki/",
            "github": "https://github.com/kikihuang123",
            "dagshub": "",
        },
    ]

    # Define social link templates - each type has a fixed position (vertical stack)
    def get_linkedin_html(url):
        if url:
            return f"""<a href="{url}" target="_blank" style="display: inline-flex; align-items: center; gap: 0.4rem; background: #0A66C2; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-decoration: none; font-size: 0.9rem; width: 120px; justify-content: center;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/></svg>LinkedIn</a>"""
        return '<div style="height: 36px; width: 120px;"></div>'

    def get_github_html(url):
        if url:
            return f"""<a href="{url}" target="_blank" style="display: inline-flex; align-items: center; gap: 0.4rem; background: #24292e; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-decoration: none; font-size: 0.9rem; width: 120px; justify-content: center;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>GitHub</a>"""
        return '<div style="height: 36px; width: 120px;"></div>'

    def get_dagshub_html(url):
        if url:
            return f"""<a href="{url}" target="_blank" style="display: inline-flex; align-items: center; gap: 0.4rem; background: #6B4FBB; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-decoration: none; font-size: 0.9rem; width: 120px; justify-content: center;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zM4.5 7.5a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1h-7zM4.5 10a.5.5 0 0 1 0-1h7a.5.5 0 0 1 0 1h-7z"/></svg>DagsHub</a>"""
        return '<div style="height: 36px; width: 120px;"></div>'

    # Build author cards HTML - using a flex container for equal height
    cards_html = '<div style="display: flex; gap: 1.5rem; align-items: stretch;">'

    for author in authors:
        # Get author image HTML (round picture or fallback emoji)
        author_image = get_author_image_html(author.get("image", ""), author["name"])

        # Build social links as vertical stack - each link type has fixed position
        linkedin_html = get_linkedin_html(author.get("linkedin", ""))
        github_html = get_github_html(author.get("github", ""))
        dagshub_html = get_dagshub_html(author.get("dagshub", ""))

        # Vertical stack of social links with fixed positions
        social_html = f'<div style="display: flex; flex-direction: column; gap: 0.5rem; align-items: center;">{linkedin_html}{github_html}{dagshub_html}</div>'

        # Build the card HTML with flex: 1 for equal width and display: flex for internal alignment
        # Use fixed heights for image and name sections to ensure perfect alignment
        cards_html += f"""<div style="flex: 1; background: {COLORS['card_bg']}; padding: 2rem; border-radius: 16px; text-align: center; border-top: 4px solid {COLORS['clinical_blue']}; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: flex; flex-direction: column; align-items: center;"><div style="height: 136px; display: flex; align-items: center; justify-content: center;">{author_image}</div><div style="height: 60px; display: flex; align-items: center; justify-content: center;"><h3 style="color: {COLORS['clinical_blue']}; margin: 0; font-size: 1.3rem; line-height: 1.3;">{author['name']}</h3></div>{social_html}</div>"""

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # ==========================================================================
    # PROJECT INFO
    # ==========================================================================

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['clinical_blue']} 0%, #264653 100%); 
                    padding: 2rem; border-radius: 16px; color: white; text-align: center; margin-top: 2rem;">
            <h3 style="margin-top: 0; color: white !important;">🎓 DataScientest Project</h3>
            <p style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 1.5rem;">
                This project was developed as part of the <strong>Data Scientist</strong> program at DataScientest.
            </p>
            <a href="https://github.com/chrmei/heartbeat_classification" target="_blank"
               style="display: inline-flex; align-items: center; gap: 0.5rem;
                      background: white; color: {COLORS['clinical_blue']}; padding: 0.75rem 1.5rem;
                      border-radius: 10px; text-decoration: none; font-weight: 600; font-size: 1rem;
                      transition: transform 0.2s, box-shadow 0.2s;"
               onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(255,255,255,0.3)';"
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                View Project Repository
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
