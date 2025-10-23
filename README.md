# Heartbeat Classification Project

## Overview

This project implements machine learning and deep learning architectures to classify cardiac signals from ECG (electrocardiogram) data. The project combines two datasets: **MIT-BIH Arrhythmia Dataset** (5 classes) and **PTB Diagnostic ECG Database** (2 classes - normal/abnormal).

**Project Difficulty:** 8/10

**Current Status:** Model testing phase completed with promising results. Preparing for advanced optimization and deep learning implementation.

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
│   └── interim/           # Feature-engineered datasets (✅ Complete)
│       ├── mitbih_train_features.csv
│       ├── mitbih_test_features.csv
│       ├── ptbdb_normal_features.csv
│       └── ptbdb_abnormal_features.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb  # ✅ Complete - EDA with 5 high-quality graphs
│   ├── 02_preprocessing.ipynb  # ✅ Complete - Preprocessing pipeline
│   ├── 03_model_testing_example_mit.ipynb  # ✅ Complete - MIT-BIH model testing
│   └── 03_model_testing_example_ptbdb.ipynb  # ✅ Complete - PTB-DB model testing
├── src/
│   ├── features/          # Feature engineering modules (✅ Complete)
│   ├── models/            # Model implementations (✅ Complete)
│   │   └── exploration_phase/  # Initial model testing
│   ├── utils/             # Utility functions (✅ Complete)
│   │   ├── preprocessing.py
│   │   ├── evaluation.py
│   │   ├── model_saver.py
│   │   └── audit_report.py
│   └── visualization/     # Visualization tools (✅ Complete)
│       ├── visualization.py
│       └── confusion_matrix.py
├── app/
│   └── streamlit_app.py   # ✅ Complete - Interactive Streamlit application
├── reports/
│   ├── DataAudit/         # ✅ Complete - Data quality reports
│   │   ├── data_audit_mitbih_test.csv
│   │   ├── data_audit_mitbih_train.csv
│   │   ├── data_audit_ptbdb_abnormal.csv
│   │   ├── data_audit_ptbdb_normal.csv
│   │   └── data_summary.txt
│   ├── 03_model_testing_results/  # ✅ Complete - Model comparison results
│   │   ├── model_comparison_with_sampling_on_best_models.csv
│   │   ├── model_comparison_without_resampling.csv
│   │   └── model_comparison_without_resampling_colorful.xlsx
│   └── figures/           # 🚧 Pending - Generated visualizations
├── docs/                  # ✅ Complete - Project documentation
│   ├── knowledge/
│   └── ProjectRequirements/
├── tests/                 # 🚧 Pending - Test suite
├── requirements.txt       # ✅ Complete
├── README.md              # ✅ Complete
└── .gitignore             # ✅ Complete
```

**Legend:** ✅ Complete | 🚧 In Progress | ⏳ Pending

## Current Features & Capabilities

### ✅ **Implemented Features**

#### 1. **Data Quality Analysis**
- Comprehensive data audit system (`src/utils/audit_report.py`)
- Automated generation of data quality reports for all datasets
- Statistical analysis of missing values, duplicates, and data types
- Memory usage and performance metrics

#### 2. **Data Preprocessing & Feature Engineering**
- Complete preprocessing pipeline (`src/utils/preprocessing.py`)
- Feature engineering with statistical and frequency domain features
- Class imbalance handling with multiple sampling techniques
- Data validation and quality checks

#### 3. **Model Testing & Evaluation**
- Comprehensive model comparison framework
- Support for multiple algorithms: XGBoost, Random Forest, SVM, Logistic Regression
- Advanced sampling techniques: SMOTE, ADASYN, RandomOverSampler, SMOTETomek
- Performance metrics: Accuracy, F1-macro, F1-weighted, Balanced Accuracy
- Model persistence and evaluation utilities (`src/utils/model_saver.py`, `src/utils/evaluation.py`)

#### 4. **Visualization Tools**
- Advanced ECG plotting utilities (`src/visualization/visualization.py`)
- Confusion matrix visualization (`src/visualization/confusion_matrix.py`)
- Support for single and multiple heartbeat visualization
- Peak detection and signal analysis capabilities
- Customizable plotting parameters and export options

#### 5. **Interactive Application**
- Complete Streamlit application (`app/streamlit_app.py`)
- Multi-page interface: Home, Data Exploration, Model Results, Live Prediction
- Interactive data exploration with class distribution analysis
- Sample ECG waveform visualization
- File upload functionality for live prediction

#### 6. **Data Management**
- All original datasets loaded and validated
- Feature-engineered datasets in `data/interim/`
- Clean data structure with proper organization
- Automated data audit reports generated
- Memory-efficient data handling

### 🚧 **Current Model Results**

**Best Performing Models (MIT-BIH Dataset):**
- **XGBoost with RandomOverSampler**: 98% accuracy, 92% F1-macro
- **XGBoost with SMOTE**: 98% accuracy, 91% F1-macro  
- **Random Forest with RandomOverSampler**: 98% accuracy, 90% F1-macro

**Key Findings:**
- XGBoost consistently outperforms other algorithms
- RandomOverSampler provides best sampling results
- All models achieve >98% accuracy on validation and test sets
- Strong performance across all 5 classes (F1-scores: 0.76-0.99)

### ⏳ **Upcoming Tasks (Before Report 2)**
- Apply GridSearchCV to best models (XGBoost, SVM, RandomForest)
- Test best models with and without extreme values removal (based on RR-Distance)
- Test PCA, baseline wandering removal and denoising on datasets
- Create deep learning model based on recent publications
- Advanced model optimization and hyperparameter tuning

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

### Step 2: Pre-Processing & Feature Engineering (Due: Oct 16) ✅ **COMPLETED**
- [x] Handle class imbalance (SMOTE, ADASYN, RandomOverSampler, SMOTETomek)
- [x] Normalize/standardize signals
- [x] Split train/validation/test sets properly
- [x] Feature extraction: statistical and frequency domain features
- [x] Complete preprocessing pipeline implementation

### Step 3: Baseline Modeling (Due: Oct 23) ✅ **COMPLETED**
- [x] **Baseline models**: Logistic Regression, Random Forest, SVM, XGBoost
- [x] Comprehensive model comparison with multiple sampling techniques
- [x] Performance evaluation and results documentation
- [x] Model persistence and evaluation utilities

### Step 4: Advanced Optimization (Due: Nov 6) 🚧 **IN PROGRESS**
- [ ] **GridSearchCV** for best models (XGBoost, SVM, RandomForest)
- [ ] **Extreme values removal** testing (based on RR-Distance)
- [ ] **Signal preprocessing** testing (PCA, baseline wandering removal, denoising)
- [ ] Advanced hyperparameter tuning
- [ ] Ensemble methods exploration

### Step 5: Deep Learning Implementation (Due: Nov 13) ⏳ **PENDING**
- [ ] **Deep Learning models**: e.g. 1D CNN, LSTM, CNN-LSTM hybrid, ResNet-like architectures
- [ ] Transfer learning exploration
- [ ] Model interpretability: SHAP values, Grad-CAM for CNN
- [ ] Advanced neural network architectures

### Step 6: Final Report & GitHub (Due: Nov 21) ⏳ **PENDING**
- [ ] Compile all reports
- [ ] Clean, document, and push code to GitHub

### Step 7: Streamlit App & Defense (Due: Dec 1) ⏳ **PENDING**
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

## Current Achievements

### 🏆 **Completed Milestones**

1. **Data Foundation** ✅
   - Complete data audit and quality analysis
   - Comprehensive preprocessing pipeline
   - Feature engineering with statistical and frequency domain features
   - Class imbalance handling with multiple sampling techniques

2. **Model Development** ✅
   - Baseline model testing with 4 algorithms
   - Comprehensive evaluation framework
   - Model comparison across multiple sampling techniques
   - Performance metrics and results documentation

3. **Application Development** ✅
   - Interactive Streamlit application
   - Multi-page interface for data exploration
   - Model results visualization
   - Live prediction functionality

4. **Results & Documentation** ✅
   - Model performance results (98% accuracy achieved)
   - Comprehensive project documentation
   - Code organization and modularity
   - Reproducible research workflow

### 📊 **Performance Highlights**
- **Best Model**: XGBoost with RandomOverSampler
- **Accuracy**: 98% on both validation and test sets
- **F1-Macro**: 92% across all classes
- **Class Performance**: F1-scores ranging from 0.76 to 0.99
- **Sampling**: RandomOverSampler provides optimal results

## Usage

### 🚀 **Quick Start**
Launch the interactive Streamlit application:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch Streamlit app
streamlit run app/streamlit_app.py
```

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
jupyter notebook notebooks/03_model_testing_example_mit.ipynb
jupyter notebook notebooks/03_model_testing_example_ptbdb.ipynb
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

### 🤖 **Model Training & Evaluation**
```bash
# Run model testing notebooks
jupyter notebook notebooks/03_model_testing_example_mit.ipynb
jupyter notebook notebooks/03_model_testing_example_ptbdb.ipynb

# Access saved models and results
ls reports/03_model_testing_results/
```

## Next Steps (Before Report 2)

### 🎯 **Immediate Priorities**

1. **GridSearchCV Optimization**
   - Apply to XGBoost, SVM, and RandomForest models
   - Optimize hyperparameters for best performance
   - Compare results with baseline models

2. **Data Quality Enhancement**
   - Test extreme values removal based on RR-Distance
   - Evaluate impact on model performance
   - Document findings and recommendations

3. **Signal Preprocessing**
   - Implement PCA for dimensionality reduction
   - Test baseline wandering removal techniques
   - Apply denoising methods and measure impact
   - Compare preprocessing approaches

4. **Deep Learning Implementation**
   - Research recent publications for state-of-the-art architectures
   - Implement 1D CNN, LSTM, and hybrid models
   - Compare with traditional ML approaches
   - Focus on interpretability and explainability

### 📈 **Expected Outcomes**
- Optimized hyperparameters for best models
- Enhanced data quality through preprocessing
- Deep learning models with competitive performance
- Comprehensive comparison of all approaches
- Ready-to-present results for Report 2

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
