"""
Page 10: Result Summary
Todo by Julia
"""

import os

import pandas as pd
import streamlit as st


def render():
    st.title("10: Result Summary")
    st.markdown("---")

    st.header("Model Comparison")

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
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Please add results.csv to the page_modules/ directory")

    st.write(
        """
    - DL models outperformed baseline models
    """
    )
