# AlphaCare Insurance Solutions - Risk & Predictive Analytics

[![CI/CD Pipeline](https://github.com/Biruk7479/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/Biruk7479/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project aims to optimize marketing strategy and discover "low-risk" targets for AlphaCare Insurance Solutions (ACIS) by analyzing historical insurance claim data from February 2014 to August 2015. The analysis focuses on identifying segments where premiums could be reduced to attract new clients while maintaining profitability.

## 🎯 Business Objectives

- Analyze historical insurance claim data to identify risk patterns
- Discover low-risk customer segments for premium optimization
- Perform A/B hypothesis testing on key risk drivers
- Build predictive models for claim severity and premium optimization
- Provide data-driven recommendations for marketing strategy

## 📊 Project Structure

```
Week-3/
├── .github/
│   └── workflows/          # CI/CD pipeline configurations
├── data/                   # Data directory (tracked by DVC)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── scripts/               # Python scripts for automation
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── visualizations.py
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── eda/              # EDA modules
│   ├── stats/            # Statistical analysis modules
│   └── models/           # ML models
├── tests/                 # Unit tests
├── models/                # Saved models
├── reports/               # Analysis reports (in .gitignore)
├── .gitignore
├── .dvc/                  # DVC configuration
├── requirements.txt       # Project dependencies
└── README.md
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- DVC (Data Version Control)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone git@github.com:Biruk7479/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git
cd Week-3
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC:**
```bash
dvc init
dvc remote add -d localstorage /path/to/local/storage
dvc pull
```

## 📈 Key Tasks

### Task 1: Exploratory Data Analysis (EDA)
- Data quality assessment and cleaning
- Descriptive statistics and distributions
- Univariate and multivariate analysis
- Outlier detection and treatment
- Visualization of key insights

**Key Findings:**
- Loss Ratio analysis by Province, VehicleType, and Gender
- Temporal trends in claim frequency and severity
- High-risk vehicle makes/models identification

### Task 2: Data Version Control (DVC)
- DVC initialization and configuration
- Local remote storage setup
- Data versioning and tracking
- Reproducible data pipeline

### Task 3: A/B Hypothesis Testing
Testing the following null hypotheses:
- H₀: No risk differences across provinces
- H₀: No risk differences between zip codes
- H₀: No significant margin difference between zip codes
- H₀: No significant risk difference between Women and Men

**Metrics:**
- Claim Frequency: Proportion of policies with at least one claim
- Claim Severity: Average amount of a claim
- Margin: TotalPremium - TotalClaims

### Task 4: Predictive Modeling
- **Claim Severity Model:** Predict TotalClaims amount
- **Premium Optimization Model:** Predict optimal premium values
- Models: Linear Regression, Random Forest, XGBoost
- Feature importance analysis using SHAP/LIME

## 📊 Data Description

The dataset contains insurance policy information with the following categories:

- **Policy Information:** UnderwrittenCoverID, PolicyID, TransactionMonth
- **Client Information:** Gender, MaritalStatus, Citizenship, Language, etc.
- **Location Data:** Country, Province, PostalCode, CrestaZones
- **Vehicle Details:** Make, Model, VehicleType, RegistrationYear, etc.
- **Plan Details:** SumInsured, CoverType, Product, ExcessSelected
- **Financial Data:** TotalPremium, TotalClaims

## 🚀 Usage

### Running EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Running Scripts
```bash
python scripts/data_loader.py
python scripts/preprocessing.py
```

### Running Tests
```bash
pytest tests/
```

## 📝 Key Insights & Findings

*(Will be populated as analysis progresses)*

1. **Loss Ratio Analysis:** 
   - Overall portfolio loss ratio: [TBD]
   - Provincial variations: [TBD]
   
2. **Risk Segmentation:**
   - Low-risk segments identified: [TBD]
   - High-risk factors: [TBD]

3. **Model Performance:**
   - Best performing model: [TBD]
   - Key predictive features: [TBD]

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for continuous integration:
- Automated testing on push/pull requests
- Code quality checks (linting, formatting)
- DVC data validation

## 📚 Documentation

- [Insurance Analytics Resources](https://www.fsrao.ca/media/11501/download)
- [A/B Testing Guide](https://medium.com/tiket-com/a-b-testing-hypothesis-testing-f9624ea5580e)
- [DVC Documentation](https://dvc.org/doc)

## 👥 Team

- **Facilitators:** Kerod, Mahbubah, Filimon
- **Project:** KAIM Week-3 Challenge
- **Organization:** AlphaCare Insurance Solutions (ACIS)

## 📅 Project Timeline

- **Challenge Start:** December 3, 2025
- **Interim Submission:** December 7, 2025, 8:00 PM UTC
- **Final Submission:** December 9, 2025, 8:00 PM UTC

## 🤝 Contributing

This is an educational project. For any questions or suggestions:
1. Create an issue
2. Submit a pull request
3. Follow conventional commits format

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- 10 Academy for providing the challenge framework
- AlphaCare Insurance Solutions for the business context
- All facilitators and mentors for their guidance

---

**Note:** This project is part of the KAIM Week-3 challenge focusing on insurance risk analytics and predictive modeling.
