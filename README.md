# Heartbeat Classification Project

## Overview

This project implements machine learning and deep learning architectures to classify cardiac signals from ECG (electrocardiogram) data. The project combines two datasets: **MIT-BIH Arrhythmia Dataset** (5 classes) and **PTB Diagnostic ECG Database** (2 classes - normal/abnormal).

**Project Difficulty:** 8/10

**Current Status:** ✅ **PROJECT COMPLETED** - All modeling phases completed with results exceeding benchmark performance.

## Key Results

Based on the final project report, our models achieved the following performance:

| Dataset | Model | Accuracy | Precision | Recall | F1 Score |
|---------|-------|----------|-----------|--------|----------|
| **MIT-BIH** | CNN8 (Optimized) | **98.51%** | 90.62% | 94.24% | 92.36% |
| **PTB** | CNN8 + Transfer Learning | **98.42%** | 97.51% | 98.64% | 98.05% |

**Benchmark Comparison:**
- MIT-BIH: Exceeded benchmark [2] by ~5% (benchmark: 93.40%)
- PTB: Exceeded benchmark [2] by ~2.5% (benchmark: 95.90%)

For detailed results and methodology, see the [Final Report](reports/renderings/03_Final%20Report.pdf).

## Datasets

- `mitbih_train.csv` (87,554 samples, 188 features including label)
- `mitbih_test.csv` (21,892 samples)
- `ptbdb_normal.csv` (4,046 samples)
- `ptbdb_abnormal.csv` (10,506 samples)

Each row represents one heartbeat segment with 187 time points + 1 label column.

## Data Quality Summary

Based on comprehensive data audit analysis:

| Dataset | Samples | Features | Missing Values | Duplicates | Memory Usage |
|---------|---------|----------|----------------|------------|--------------|
| **MIT-BIH Train** | 87,554 | 188 | 0 | 0 | 131.7 MB |
| **MIT-BIH Test** | 21,892 | 188 | 0 | 0 | 32.9 MB |
| **PTB Normal** | 4,046 | 188 | 0 | 1 | 6.1 MB |
| **PTB Abnormal** | 10,506 | 188 | 0 | 6 | 15.8 MB |

**Key Findings:**
- ✅ **No missing values** in any dataset
- ✅ **Clean data structure** with consistent 188 features (187 ECG samples + 1 label)
- ✅ **Minimal duplicates** (removed during preprocessing)
- ✅ **Memory efficient** data storage
- ⚠️ **Class imbalance** present in both datasets (addressed with SMOTE sampling)

## Project Structure

```
heartbeat_classification/
├── data/
│   ├── original/          # Raw Kaggle data (✅ Complete)
│   │   ├── mitbih_train.csv
│   │   ├── mitbih_test.csv
│   │   ├── ptbdb_normal.csv
│   │   └── ptbdb_abnormal.csv
│   ├── processed/         # Cleaned & preprocessed data (✅ Complete)
│   │   ├── mitbih/
│   │   └── ptb/
│   └── interim/           # Feature-engineered datasets (✅ Complete)
│       ├── mitbih_train_features.csv
│       ├── mitbih_test_features.csv
│       ├── ptbdb_normal_features.csv
│       └── ptbdb_abnormal_features.csv
├── notebooks/             # Analysis notebooks (✅ Complete)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_A_*.ipynb       # MIT-BIH baseline models
│   ├── 03_B_*.ipynb       # PTB baseline models
│   ├── 04_A_*.ipynb       # MIT-BIH deep learning models
│   ├── 04_B_*.ipynb       # PTB deep learning models
│   ├── 05_A_DL_SHAP.ipynb # MIT-BIH interpretability
│   ├── 05_B_DL_SHAP.ipynb # PTB interpretability
│   └── archive/           # Archived development notebooks
│       ├── christian/
│       └── julia/
├── src/
│   ├── utils/             # Utility functions (✅ Complete)
│   │   ├── preprocessing.py
│   │   ├── evaluation.py
│   │   ├── model_saver.py
│   │   ├── audit_report.py
│   │   └── dl_architectures.py
│   └── visualization/     # Visualization tools (✅ Complete)
│       ├── visualization.py
│       └── confusion_matrix.py
├── models/                 # Saved trained models (✅ Complete)
│   ├── MIT_02_01_baseline_models_randomized_search_no_sampling/
│   ├── MIT_02_02_baseline_models_randomized_search_sampling/
│   ├── MIT_02_03_dl_models/
│   └── PTB_04_02_dl_models/
├── reports/
│   ├── data_audit/         # Data quality reports (✅ Complete)
│   ├── baseline_models/    # Baseline model results (✅ Complete)
│   │   ├── MIT_02_01_RANDOMIZED_SEARCH/
│   │   └── MIT_02_02_RS_SAMPLING/
│   ├── deep_learning/      # Deep learning model results (✅ Complete)
│   │   ├── cnn8_transfer/
│   │   ├── models_optimization/
│   │   └── model_comparison.csv
│   ├── interpretability/   # SHAP analysis results (✅ Complete)
│   │   ├── SHAP_MIT/
│   │   └── SHAP_PTB/
│   └── renderings/         # Project reports (✅ Complete)
│       ├── 01_Rendering 1.pdf
│       ├── 02_Rendering2-Report.pdf
│       └── 03_Final Report.pdf
├── docs/                  # Project documentation (✅ Complete)
│   ├── knowledge/
│   └── ProjectRequirements/
├── tests/                 # Test suite
├── requirements.txt       # Dependencies (✅ Complete)
├── pyproject.toml         # Project configuration (✅ Complete)
├── README.md              # This file (✅ Complete)
└── CONTRIBUTING.md        # Contribution guidelines (✅ Complete)
```

**Legend:** ✅ Complete

## Notebook Organization

The project notebooks follow a systematic numbering scheme:

- **01-02**: Data exploration and preprocessing
- **03_A**: MIT-BIH baseline models (RandomizedSearch, GridSearch, evaluation)
- **03_B**: PTB baseline models (LazyClassifier, GridSearch, evaluation)
- **04_A**: MIT-BIH deep learning models (CNN, DNN, LSTM, optimization)
- **04_B**: PTB deep learning models (Transfer learning)
- **05_A/B**: Model interpretability (SHAP analysis)

For detailed notebook documentation, see [notebooks/README.md](notebooks/README.md).

**Archived Notebooks:**
Development notebooks from earlier iterations are preserved in `notebooks/archive/` for reference:
- `archive/christian/`: Early development notebooks
- `archive/julia/`: Alternative approaches and experiments

## Features & Capabilities

### ✅ **Implemented Features**

#### 1. **Data Quality Analysis**
- Comprehensive data audit system (`src/utils/audit_report.py`)
- Automated generation of data quality reports for all datasets
- Statistical analysis of missing values, duplicates, and data types
- Memory usage and performance metrics

#### 2. **Data Preprocessing & Feature Engineering**
- Complete preprocessing pipeline (`src/utils/preprocessing.py`)
- Feature engineering with statistical and frequency domain features
- Class imbalance handling with SMOTE (selected as optimal method)
- Data validation and quality checks
- Duplicate removal for PTB dataset

#### 3. **Baseline Model Development**
- Comprehensive model comparison framework
- Multiple algorithms: XGBoost, Random Forest, SVM, Logistic Regression, KNN, Decision Tree, LDA, ANN
- RandomizedSearch and GridSearch hyperparameter optimization
- Performance metrics: Accuracy, Precision, Recall, F1-score
- Model persistence and evaluation utilities

#### 4. **Deep Learning Models**
- CNN architectures (inspired by Kachuee et al. 2018)
- DNN and LSTM models
- Transfer learning from MIT-BIH to PTB dataset
- Model optimization with dropout and batch normalization
- Training on Google Colab with GPU acceleration

#### 5. **Model Interpretability**
- SHAP (SHapley Additive exPlanations) analysis
- Feature importance visualization
- Decision pattern analysis for both datasets

#### 6. **Visualization Tools**
- Advanced ECG plotting utilities (`src/visualization/visualization.py`)
- Confusion matrix visualization (`src/visualization/confusion_matrix.py`)
- Support for single and multiple heartbeat visualization
- Peak detection and signal analysis capabilities
- Customizable plotting parameters and export options

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd heartbeat_classification
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
Place the original data files in `data/original/` directory:
- mitbih_train.csv
- mitbih_test.csv
- ptbdb_normal.csv
- ptbdb_abnormal.csv

**Note:** The datasets are available from Kaggle. Please ensure you have proper access and licensing.

## Usage Examples

### 📊 **Data Analysis**
Generate comprehensive data audit reports:

```bash
# Run data audit analysis
python -c "from src.utils.audit_report import generate_data_audit_report; generate_data_audit_report()"
```

### 📓 **Jupyter Notebooks**
Explore the data and models interactively:

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb

# Baseline models
jupyter notebook notebooks/03_A_02_01_baseline_models_randomized_search.ipynb
jupyter notebook notebooks/03_B_02_baseline_models_lazy_classifier.ipynb

# Deep learning models
jupyter notebook notebooks/04_A_02_CNN_models_smote.ipynb
jupyter notebook notebooks/04_B_01_CNN_Transfer.ipynb

# Interpretability
jupyter notebook notebooks/05_A_DL_SHAP.ipynb
jupyter notebook notebooks/05_B_DL_SHAP.ipynb
```

### 🔧 **Development**
For development and testing:

```bash
# Run data audit
python src/utils/audit_report.py

# Test visualization utilities
python -c "from src.visualization.visualization import plot_heartbeat; import numpy as np; plot_heartbeat(np.random.randn(187))"

# Run model evaluation
python src/utils/evaluation.py
```

## Project Timeline

### Step 1: Data Mining & Visualization ✅ **COMPLETED**
- [x] Create project structure
- [x] Set up requirements.txt
- [x] Write README
- [x] Load and explore datasets
- [x] Generate data audit reports with comprehensive analysis
- [x] Document data quality issues and class imbalance
- [x] Create visualization utilities for ECG plotting

### Step 2: Pre-Processing & Feature Engineering ✅ **COMPLETED**
- [x] Handle class imbalance (SMOTE selected as optimal method)
- [x] Normalize/standardize signals
- [x] Split train/validation/test sets properly
- [x] Feature extraction: statistical and frequency domain features
- [x] Complete preprocessing pipeline implementation
- [x] Duplicate removal for PTB dataset

### Step 3: Baseline Modeling ✅ **COMPLETED**
- [x] **Baseline models**: Multiple algorithms tested
- [x] RandomizedSearch for initial model comparison
- [x] GridSearch for hyperparameter optimization
- [x] Comprehensive model comparison with SMOTE sampling
- [x] Performance evaluation and results documentation
- [x] Model persistence and evaluation utilities

### Step 4: Advanced Optimization ✅ **COMPLETED**
- [x] **GridSearchCV** for best models
- [x] **Extreme values analysis** (RR-Distance analysis)
- [x] Advanced hyperparameter tuning
- [x] Model selection and final evaluation

### Step 5: Deep Learning Implementation ✅ **COMPLETED**
- [x] **Deep Learning models**: CNN, DNN, LSTM architectures
- [x] Transfer learning from MIT-BIH to PTB dataset
- [x] Model optimization with dropout and batch normalization
- [x] Model interpretability: SHAP values analysis
- [x] Advanced neural network architectures

### Step 6: Final Report & Documentation ✅ **COMPLETED**
- [x] Compile all reports
- [x] Clean, document, and organize code
- [x] Final report generation
- [x] Results documentation and comparison with benchmark

## Key Considerations

**Metrics:** Focus on F1-score, precision, recall (due to class imbalance). Also track accuracy, confusion matrix.

**Challenges Addressed:**
- ✅ Severe class imbalance in both datasets (solved with SMOTE)
- ✅ High dimensionality (187 features) - handled by deep learning architectures
- ✅ Two separate datasets with different objectives (5-class vs 2-class)
- ✅ Time series nature - handled with appropriate architectures
- ✅ Model interpretability - addressed with SHAP analysis

**Success Criteria:**
- ✅ Beat benchmark performance (exceeded by 2.5-5%)
- ✅ Robust model with good generalization
- ✅ Clear, professional reports with business insights
- ✅ Model interpretability for clinical validation

## Results Summary

### 🏆 **Final Model Performance**

**MIT-BIH Arrhythmia Classification (5 classes):**
- **Best Model**: CNN8 (Optimized with dropout and batch normalization)
- **Accuracy**: 98.51%
- **Precision**: 90.62%
- **Recall**: 94.24%
- **F1 Score**: 92.36%

**PTB Myocardial Infarction Detection (2 classes):**
- **Best Model**: CNN8 with Transfer Learning (last residual block unfrozen)
- **Accuracy**: 98.42%
- **Precision**: 97.51%
- **Recall**: 98.64%
- **F1 Score**: 98.05%

### 📊 **Key Achievements**

1. **Exceeded Benchmark Performance**
   - MIT-BIH: +5% improvement over benchmark [2]
   - PTB: +2.5% improvement over benchmark [2]

2. **Robust Model Development**
   - Comprehensive baseline model comparison
   - Advanced deep learning architectures
   - Transfer learning implementation

3. **Model Interpretability**
   - SHAP analysis for both datasets
   - Feature importance visualization
   - Clinically relevant pattern identification

4. **Complete Documentation**
   - Comprehensive data audit
   - Detailed model evaluation reports
   - Final project report with methodology and results

## Bibliography

Key research articles referenced in this project:

1. **Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018)**. ECG Heartbeat Classification: A Deep Transferable Representation. CoRR. doi: 10.48550/arXiv.1805.00794

2. **Murat, F., Yildirim, O., Talo, M., Baloglu, U. B., Demir, Y., & Acharya, U. R. (2020)**. Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review. Computers in Biology and Medicine. doi:10.1016/j.compbiomed.2020.103726

3. **Ansari, Y., Mourad, O., Qaraqe, K., & Serpedin, E. (2023)**. Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023. doi: 10.3389/fphys.2023.1246746

For complete bibliography, see the [Final Report](reports/renderings/03_Final%20Report.pdf).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is part of the DataScientest Data Scientist training program.

## Contact

For questions about this project, please refer to the DataScientest training materials and instructor support.

---

**Project Team:** Christian Meister, Julia Schmidt, Tzu-Jung Huang  
**Completion Date:** November 2025
