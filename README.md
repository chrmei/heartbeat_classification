# Heartbeat Classification Project

## Overview

This project implements deep neural network architectures to classify cardiac signals from ECG (electrocardiogram) data. The project combines two datasets: **MIT-BIH Arrhythmia Dataset** (5 classes) and **PTB Diagnostic ECG Database** (2 classes - normal/abnormal).

**Project Difficulty:** 8/10

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
- ⚠️ **Class imbalance** present in both datasets (expected for medical data)

## Project Structure

```
heartbeat_classification/
├── data/
│   ├── original/          # Raw Kaggle data (✅ Complete)
│   │   ├── mitbih_train.csv
│   │   ├── mitbih_test.csv
│   │   ├── ptbdb_normal.csv
│   │   └── ptbdb_abnormal.csv
│   ├── processed/         # Cleaned & preprocessed data (🚧 Pending)
│   └── interim/           # Intermediate data transformations (🚧 Pending)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # ✅ In Progress
│   ├── 02_preprocessing_feature_engineering.ipynb  # 🚧 Pending
│   ├── 03_baseline_modeling.ipynb  # 🚧 Pending
│   ├── 04_advanced_modeling.ipynb  # 🚧 Pending
│   └── 05_deep_learning.ipynb  # 🚧 Pending
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── audit_report.py  # ✅ Complete - Data quality analysis
│   ├── features/  # 🚧 Pending
│   ├── models/  # 🚧 Pending
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualization.py  # ✅ Complete - ECG plotting utilities
│   └── utils/  # 🚧 Pending
├── app/  🚧 Pending streamlit app
├── models/
│   └── saved_models/       # 🚧 Pending - Trained model artifacts
├── reports/
│   ├── DataAudit/          # ✅ Complete - Data quality reports
│   │   ├── data_audit_mitbih_test.csv
│   │   ├── data_audit_mitbih_train.csv
│   │   ├── data_audit_ptbdb_abnormal.csv
│   │   ├── data_audit_ptbdb_normal.csv
│   │   └── data_summary.txt
│   └── figures/  # 🚧 Pending
├── docs/  # ✅ Complete - Project documentation
│   ├── knowledge/
│   └── ProjectRequirements/
├── tests/  # 🚧 Pending
├── requirements.txt  # ✅ Complete
├── README.md  # ✅ Complete
└── .gitignore  # ✅ Complete
```

**Legend:** ✅ Complete | 🚧 In Progress | ⏳ Pending

## Current Features & Capabilities

### ✅ **Implemented Features**

#### 1. **Data Quality Analysis**
- Comprehensive data audit system (`src/data/audit_report.py`)
- Automated generation of data quality reports for all datasets
- Statistical analysis of missing values, duplicates, and data types
- Memory usage and performance metrics

#### 2. **Visualization Tools**
- Advanced ECG plotting utilities (`src/visualization/visualization.py`)
- Support for single and multiple heartbeat visualization
- Peak detection and signal analysis capabilities
- Customizable plotting parameters and export options

#### 3. **Data Management**
- All original datasets loaded and validated
- Clean data structure with proper organization
- Automated data audit reports generated
- Memory-efficient data handling

### 🚧 **In Development**
- Data preprocessing and feature engineering pipelines
- Model training and evaluation frameworks
- Advanced visualization and reporting tools
- Interactive Streamlit application

### ⏳ **Planned Features**
- Machine learning model implementations
- Model interpretability and SHAP analysis
- Automated report generation
- Performance optimization and scaling

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

## Project Timeline

### Step 1: Data Mining & Visualization (Due: Oct 2) ✅ **COMPLETED**
- [x] Create project structure
- [x] Set up requirements.txt
- [x] Write README
- [x] Load and explore datasets
- [x] Generate data audit reports with comprehensive analysis
- [x] Document data quality issues and class imbalance
- [x] Create visualization utilities for ECG plotting

### Step 2: Pre-Processing & Feature Engineering (Due: Oct 16) 🚧 **IN PROGRESS**
- [ ] Handle class imbalance (SMOTE, class weights, undersampling)
- [ ] Normalize/standardize signals
- [ ] Split train/validation/test sets properly
- [ ] Consider feature extraction: wavelet transforms, FFT, statistical features

### Step 3: Modeling (Due: Nov 13) ⏳ **PENDING**
- **Baseline models** (Oct 23): Logistic Regression, Random Forest, SVM
- **Optimization** (Nov 6): Grid Search, Cross-Validation, ensemble methods
- **Deep Learning** (Nov 13): 1D CNN, LSTM, CNN-LSTM hybrid, ResNet-like architectures
- Transfer learning exploration
- Model interpretability: SHAP values, Grad-CAM for CNN


### Step 4: Final Report & GitHub (Due: Nov 21) ⏳ **PENDING**
- Compile all reports
- Clean, document, and push code to GitHub

### Step 5: Streamlit App & Defense (Due: Dec 1) ⏳ **PENDING**
- [ ] Interactive app with multiple tabs (EDA, Model Results, Live Prediction)
- [ ] 20-minute presentation + 10-minute Q&A

## Key Considerations

**Metrics:** Focus on F1-score, precision, recall (due to class imbalance). Also track accuracy, confusion matrix, ROC-AUC.

**Challenges:**
- Severe class imbalance in both datasets
- High dimensionality (187 features)
- Two separate datasets with different objectives (5-class vs 2-class)
- Time series nature requires special handling

**Success Criteria:**
- Beat benchmark performance (research existing Kaggle solutions)
- Robust model with good generalization
- Clear, professional reports with business insights
- Working Streamlit deployment

## Usage

### 📊 **Data Analysis**
Generate comprehensive data audit reports:

```bash
# Run data audit analysis
python -c "from src.data.audit_report import generate_data_audit_report; generate_data_audit_report()"
```

### 📓 **Jupyter Notebooks**
Explore the data interactively:

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 🔧 **Development**
For development and testing:

```bash
# Run data audit
python src/data/audit_report.py

# Test visualization utilities
python -c "from src.visualization.visualization import plot_heartbeat; import numpy as np; plot_heartbeat(np.random.randn(187))"
```

### 🤖 **Model Training** (Coming Soon)
```bash
# This will be available after model implementation
python src/models/train.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the DataScientest Data Scientist training program.

## Contact

For questions about this project, please refer to the DataScientest training materials and instructor support.
