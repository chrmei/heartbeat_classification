# Heartbeat Classification - Streamlit App

Interactive presentation app for the ECG Heartbeat Classification project, showcasing deep learning models for arrhythmia detection and myocardial infarction diagnosis.

## Features

- **13 Interactive Pages**: Complete scientific presentation from introduction to conclusion
- **Live Model Predictions**: Test XGBoost and CNN models on real ECG samples
- **SHAP Interpretability**: Explore feature importance and model decisions
- **Professional Design**: Custom medical-themed UI with responsive layouts

## Quick Start

### Local Development

```bash
# From project root
cd app
streamlit run app.py

# Or from project root directly
streamlit run app/app.py
```

The app opens at `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub**: Ensure the repository is on GitHub

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `chrmei/heartbeat_classification`
   - Main file path: `app/app.py`
   - Click "Deploy"

3. **Requirements**: The app uses `app/requirements.txt` (optimized for cloud deployment)

## Project Structure

```
app/
├── app.py                    # Main entry point
├── requirements.txt          # Deployment dependencies
├── .streamlit/
│   └── config.toml          # Theme and server configuration
├── page_modules/
│   ├── __init__.py
│   ├── styles.py            # Shared CSS and styling utilities
│   ├── state_utils.py       # Session state management
│   ├── page_1_introduction.py
│   ├── page_2_data_overview.py
│   ├── page_3_preprocessing.py
│   ├── page_4_general_modeling_overview.py
│   ├── page_5_baseline_mit.py
│   ├── page_6_baseline_ptb.py
│   ├── page_7_dl_models.py
│   ├── page_8_dl_mit.py
│   ├── page_9_dl_ptb.py
│   ├── page_10_summary.py
│   ├── page_11_shap_mit.py
│   ├── page_12_shap_ptb.py
│   └── page_13_conclusion.py
├── images/                   # Static images and figures
├── tables/                   # CSV data for results tables
└── README.md
```

## Page Overview

| Page | Title | Description |
|------|-------|-------------|
| 1 | Introduction | Project context, motivation, and goals |
| 2 | Data Overview | MIT-BIH and PTB dataset exploration |
| 3 | Preprocessing | RR-distance analysis and data quality |
| 4 | Modeling Overview | General approach and methodology |
| 5 | Baseline MIT | XGBoost results on MIT-BIH dataset |
| 6 | Baseline PTB | XGBoost results on PTB dataset |
| 7 | DL Architecture | CNN and LSTM model designs |
| 8 | DL MIT | Deep learning results on MIT-BIH |
| 9 | DL PTB | Transfer learning results on PTB |
| 10 | Summary | Performance comparison and key findings |
| 11 | SHAP MIT | Interpretability analysis for MIT-BIH |
| 12 | SHAP PTB | Interpretability analysis for PTB |
| 13 | Conclusion | Clinical implications and future work |

## Key Results

| Metric | MIT-BIH | PTB |
|--------|---------|-----|
| Accuracy | 98.51% | 98.42% |
| F1-Score | 0.9236 | 0.98 |
| Benchmark | 93.4% (Kachuee 2018) | 95.9% (Kachuee 2018) |
| Improvement | +5.11% | +2.52% |

## Development Notes

### Adding New Pages

1. Create `page_modules/page_X_name.py` with a `render()` function
2. Add page to `NAV_SECTIONS` in `app.py`
3. Add routing logic in the page routing section

### Styling

- Use `from page_modules.styles import ...` for consistent styling
- Custom CSS is injected via `inject_custom_css()`
- Matplotlib styling: `apply_matplotlib_style()`
- Color palette available in `COLORS` dict

### State Management

- Use `from page_modules.state_utils import ...` for session state
- Dataclass-based state management reduces boilerplate
- Each page has isolated state to prevent conflicts

## Dependencies

See `requirements.txt` for deployment dependencies. Key packages:
- `streamlit>=1.28.0`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`
- `tensorflow>=2.13.0` (for model loading)
- `shap>=0.44.0`

## Team

- **Kiki**: Introduction, Data Overview, Preprocessing, Conclusion
- **Christian**: Baseline Models (pages 4-6)
- **Julia**: Deep Learning Models, SHAP Analysis (pages 7-12)

## License

See project root LICENSE file.

---

**Repository**: [github.com/chrmei/heartbeat_classification](https://github.com/chrmei/heartbeat_classification)
