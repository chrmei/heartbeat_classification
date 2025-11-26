# Heartbeat Classification - Streamlit Presentation App

This Streamlit app provides an interactive presentation of the heartbeat classification project results.

## Structure

- `app.py` - Main entry point for the Streamlit application
- `page_modules/` - Directory containing all page modules (renamed from `pages/` to avoid Streamlit auto-detection)
  - `page_1_introduction.py` - Introduction (Todo by Kiki)
  - `page_2_data_overview.py` - Data overview (Todo by Kiki)
  - `page_3_preprocessing.py` - Pre-processing RR-distance analysis (Todo by Kiki)
  - `page_4_baseline_models.py` - Baseline models overview (Todo by Christian)
  - `page_5_baseline_mit.py` - Baseline results MIT (Todo by Christian)
  - `page_6_baseline_ptb.py` - Baseline results PTB (Todo by Christian)
  - `page_7_dl_mit.py` - Deep Learning MIT (Todo by Julia)
  - `page_8_dl_ptb.py` - Deep Learning PTB Transfer (Todo by Julia)
  - `page_9_summary.py` - Result summary (Todo by Julia)
  - `page_10_shap_mit.py` - SHAP analysis MIT (Todo by Julia)
  - `page_11_shap_ptb.py` - SHAP analysis PTB (Todo by Julia)
  - `page_12_conclusion.py` - Conclusion (Todo by Kiki)

## Running the App

From the project root directory, run:

```bash
streamlit run app/app.py
```

Or from the `app/` directory:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Current Status

This is a base template with placeholders. Each page contains:
- Titles and subtitles
- Comments indicating what needs to be implemented
- TODO assignments for each team member

## Next Steps

Each team member should implement their assigned pages:
- **Kiki**: Pages 1, 2, 3, 12
- **Christian**: Pages 4, 5, 6
- **Julia**: Pages 7, 8, 9, 10, 11

## Notes

- All pages are currently placeholders with structure and TODO comments
- Data loading, model loading, and visualization functions need to be implemented
- The sidebar navigation is functional and allows switching between pages
- Make sure to load models and data efficiently to avoid long loading times during presentation

