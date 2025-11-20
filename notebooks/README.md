# Notebooks Documentation

This directory contains all Jupyter notebooks used for data exploration, preprocessing, model development, and analysis in the Heartbeat Classification project.

## Notebook Numbering Scheme

The notebooks follow a systematic numbering scheme that reflects the project workflow:

### Phase 1: Data Exploration & Preprocessing
- **`01_data_exploration.ipynb`**: Initial exploratory data analysis (EDA) with comprehensive visualizations
- **`02_preprocessing.ipynb`**: Data preprocessing pipeline, feature engineering, and data quality checks

### Phase 2: Baseline Models - MIT-BIH Dataset (03_A_*)
- **`03_A_02_01_baseline_models_randomized_search.ipynb`**: Initial baseline model testing without sampling
- **`03_A_02_02_baseline_models_randomized_search_sampling.ipynb`**: Baseline models with various sampling techniques (SMOTE, ADASYN, etc.)
- **`03_A_03_01_baseline_models_grid_search.ipynb`**: Hyperparameter optimization using GridSearchCV
- **`03_A_04_baseline_models_final_model_eval.ipynb`**: Final evaluation of optimized baseline models

### Phase 2: Baseline Models - PTB Dataset (03_B_*)
- **`03_B_02_baseline_models_lazy_classifier.ipynb`**: Initial baseline model testing using LazyClassifier
- **`03_B_03_baseline_models_grid_search.ipynb`**: Hyperparameter optimization using GridSearchCV
- **`03_B_04_baseline_models_final_model_eval.ipynb`**: Final evaluation of optimized baseline models

### Phase 3: Deep Learning - MIT-BIH Dataset (04_A_*)
- **`04_A_02_CNN_models_smote.ipynb`**: Convolutional Neural Network (CNN) models with SMOTE sampling
- **`04_A_02_DNN_models_smote.ipynb`**: Deep Neural Network (DNN) models with SMOTE sampling
- **`04_A_02_LSTM_models_smote.ipynb`**: Long Short-Term Memory (LSTM) models with SMOTE sampling
- **`04_A_03_CNN_optimization.ipynb`**: CNN architecture optimization (dropout, batch normalization, learning rate tuning)

### Phase 3: Deep Learning - PTB Dataset (04_B_*)
- **`04_B_01_CNN_Transfer.ipynb`**: Transfer learning from MIT-BIH CNN model to PTB dataset

### Phase 4: Model Interpretability (05_*)
- **`05_A_DL_SHAP.ipynb`**: SHAP (SHapley Additive exPlanations) analysis for MIT-BIH deep learning models
- **`05_B_DL_SHAP.ipynb`**: SHAP analysis for PTB deep learning models

## Workflow Overview

```
01_data_exploration.ipynb
    ↓
02_preprocessing.ipynb
    ↓
┌─────────────────────────┬─────────────────────────┐
│   MIT-BIH Dataset       │    PTB Dataset          │
│   (03_A_*)              │    (03_B_*)             │
│                         │                         │
│ 03_A_02_01: Randomized │ 03_B_02: LazyClassifier│
│ 03_A_02_02: + Sampling  │ 03_B_03: GridSearch     │
│ 03_A_03_01: GridSearch │ 03_B_04: Final Eval     │
│ 03_A_04: Final Eval     │                         │
└─────────────────────────┴─────────────────────────┘
    ↓                           ↓
┌─────────────────────────┬─────────────────────────┐
│   MIT-BIH DL Models     │    PTB DL Models        │
│   (04_A_*)              │    (04_B_*)             │
│                         │                         │
│ 04_A_02: CNN/DNN/LSTM   │ 04_B_01: Transfer       │
│ 04_A_03: Optimization   │                         │
└─────────────────────────┴─────────────────────────┘
    ↓                           ↓
┌─────────────────────────┬─────────────────────────┐
│   MIT-BIH SHAP          │    PTB SHAP             │
│   (05_A_*)              │    (05_B_*)             │
│                         │                         │
│ 05_A_DL_SHAP            │ 05_B_DL_SHAP            │
└─────────────────────────┴─────────────────────────┘
```

## Dataset-Specific Notation

- **A** suffix: MIT-BIH Arrhythmia Dataset (5 classes: Normal, Supraventricular, Ventricular, Fusion, Unknown)
- **B** suffix: PTB Diagnostic ECG Database (2 classes: Normal, Abnormal/MI)

## Key Notebooks by Task

### Data Exploration
- Start with `01_data_exploration.ipynb` to understand the dataset structure and class distributions

### Preprocessing
- `02_preprocessing.ipynb` contains the complete preprocessing pipeline

### Baseline Model Development
- **MIT-BIH**: Start with `03_A_02_01_baseline_models_randomized_search.ipynb`
- **PTB**: Start with `03_B_02_baseline_models_lazy_classifier.ipynb`

### Deep Learning Models
- **MIT-BIH**: `04_A_02_CNN_models_smote.ipynb` for CNN models
- **PTB**: `04_B_01_CNN_Transfer.ipynb` for transfer learning approach

### Model Interpretability
- `05_A_DL_SHAP.ipynb` and `05_B_DL_SHAP.ipynb` for understanding model decisions

## Archived Notebooks

The `archive/` directory contains development notebooks from earlier project iterations:

### `archive/christian/`
Early development notebooks including:
- Initial data exploration and preprocessing
- Early baseline model experiments
- CNN model development iterations (CNN1-9)
- PTB-specific grid search experiments

### `archive/julia/`
Alternative approaches and experiments including:
- Different sampling strategies
- Alternative model architectures
- Early SHAP analysis attempts
- CNN-LSTM hybrid architectures

**Note:** These archived notebooks are preserved for reference but may not reflect the final methodology. Refer to the main notebooks for the final implementation.

## Running Notebooks

### Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure data files are in `data/original/` directory
3. Run preprocessing notebook first to generate intermediate data files

### Execution Order
For a complete analysis, notebooks should be run in the following order:

1. **Data Exploration & Preprocessing**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   jupyter notebook notebooks/02_preprocessing.ipynb
   ```

2. **Baseline Models** (can run in parallel for different datasets)
   ```bash
   # MIT-BIH
   jupyter notebook notebooks/03_A_02_01_baseline_models_randomized_search.ipynb
   jupyter notebook notebooks/03_A_02_02_baseline_models_randomized_search_sampling.ipynb
   jupyter notebook notebooks/03_A_03_01_baseline_models_grid_search.ipynb
   jupyter notebook notebooks/03_A_04_baseline_models_final_model_eval.ipynb
   
   # PTB
   jupyter notebook notebooks/03_B_02_baseline_models_lazy_classifier.ipynb
   jupyter notebook notebooks/03_B_03_baseline_models_grid_search.ipynb
   jupyter notebook notebooks/03_B_04_baseline_models_final_model_eval.ipynb
   ```

3. **Deep Learning Models**
   ```bash
   # MIT-BIH (run CNN optimization after initial CNN/DNN/LSTM)
   jupyter notebook notebooks/04_A_02_CNN_models_smote.ipynb
   jupyter notebook notebooks/04_A_02_DNN_models_smote.ipynb
   jupyter notebook notebooks/04_A_02_LSTM_models_smote.ipynb
   jupyter notebook notebooks/04_A_03_CNN_optimization.ipynb
   
   # PTB (requires trained MIT-BIH model)
   jupyter notebook notebooks/04_B_01_CNN_Transfer.ipynb
   ```

4. **Interpretability**
   ```bash
   jupyter notebook notebooks/05_A_DL_SHAP.ipynb
   jupyter notebook notebooks/05_B_DL_SHAP.ipynb
   ```

## Output Locations

### Models
- Baseline models: `models/MIT_02_*/` and `models/PTB_*/`
- Deep learning models: `models/MIT_02_03_dl_models/` and `models/PTB_04_02_dl_models/`

### Reports
- Baseline model results: `reports/baseline_models/`
- Deep learning results: `reports/deep_learning/`
- SHAP analysis: `reports/interpretability/`

### Data
- Processed data: `data/processed/`
- Feature-engineered data: `data/interim/`

## Notes

- **SMOTE Sampling**: Selected as the optimal sampling technique after comparison with RandomOverSampler, ADASYN, and SMOTETomek
- **Transfer Learning**: The PTB model uses transfer learning from the MIT-BIH CNN model, with the last residual block unfrozen for fine-tuning
- **Model Selection**: CNN8 architecture (inspired by Kachuee et al. 2018) achieved best performance after optimization
- **GPU Training**: Deep learning models were trained on Google Colab with GPU acceleration for efficiency

## References

For detailed methodology and results, see:
- [Final Report](../reports/renderings/03_Final%20Report.pdf)
- [Main README](../README.md)

For questions or issues with specific notebooks, refer to the inline documentation and comments within each notebook.

