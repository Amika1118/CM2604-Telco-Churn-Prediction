```markdown
# CM2604 Machine Learning Coursework: Telco Customer Churn Prediction

## Project Overview
This repository contains a complete machine learning solution for predicting customer churn in the telecommunications industry. Implemented as part of the CM2604 Machine Learning coursework, this project demonstrates end-to-end data science workflow from exploratory data analysis to model deployment with ethical considerations.

### Coursework Requirements
This project addresses all coursework specifications:
- Task 1: Comprehensive Exploratory Data Analysis (15%)
- Task 2: Model implementation with Decision Tree and Neural Network (25% each for corpus preparation, implementation, results)
- Task 3: AI ethics framework and deployment strategy (10%)

### Business Problem
Predict which customers are likely to cancel their subscriptions ("churn") based on demographic information, service usage, and billing details. This enables proactive retention strategies to reduce revenue loss.

## Repository Structure

```
CM2604-Telco-Churn-Prediction/
├── data/                           # Dataset storage
│   └── dataset.csv                # Processed dataset (not included, upload separately)
├── explainability/                 # Model explainability outputs
│   └── shap_beeswarm.png          # SHAP analysis visualization
├── models/                         # Trained machine learning models
│   ├── dt_baseline.pkl            # Baseline Decision Tree
│   ├── dt_tuned.pkl               # Manually tuned Decision Tree
│   ├── nn_baseline.h5             # Baseline Neural Network
│   └── nn_tuned.h5                # Manually tuned Neural Network
├── plots/                          # All visualization outputs
│   ├── churn_distribution.png     # EDA: Target variable distribution
│   ├── numerical_distributions.png # EDA: Numerical features
│   ├── correlation_matrix.png     # EDA: Feature correlations
│   ├── dt_baseline_roc.png        # Decision Tree ROC curves
│   ├── dt_tuned_roc.png
│   ├── nn_baseline_roc.png
│   └── nn_tuned_roc.png
├── preprocessing/                  # Data preprocessing components
│   └── preprocessor.pkl           # Fitted preprocessing pipeline
├── reports/                        # Evaluation results and documentation
│   ├── eda_insights.txt           # Key insights from EDA
│   ├── dt_baseline.txt            # Decision Tree baseline results
│   ├── dt_tuned.txt               # Decision Tree tuned results
│   ├── nn_baseline.txt            # Neural Network baseline results
│   ├── nn_tuned.txt               # Neural Network tuned results
│   ├── model_comparison.csv       # Model comparison (CSV)
│   ├── model_comparison.txt       # Model comparison (text)
│   ├── ethical_framework.txt      # AI ethics documentation
│   ├── deployment_strategy.txt    # Post-deployment strategy
│   └── bias_analysis.txt          # Fairness considerations
├── CM2604-Telco-Churn-Prediction.ipynb          # Main project notebook
├── CM2604_Telco_Churn_Prediction (1).ipynb      # Alternative notebook version
├── [.ipynb file]                                # Python script 
├── [.py file]                                   # Python script
└── README.md                                    # Project documentation (this file)
```

## Recent Updates (December 2025)

### Added Files:
1. **CM2604_Telco_Churn_Prediction (1).ipynb** - Alternative notebook version with additional experiments
2. **Python Script Files** - Two Python scripts for modular code organization

### Current Status:
- All coursework requirements completed and documented
- Multiple notebook versions available for comparison
- Modular Python scripts for reusable components
- Ready for final submission and viva preparation

## Quick Start (Google Colab)

### 1. Upload and Run the Notebook
1. Open Google Colab (https://colab.research.google.com/)
2. Upload `CM2604-Telco-Churn-Prediction.ipynb` 
3. Run cells sequentially from top to bottom

### 2. Dataset Upload
When prompted, upload the Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) from Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### 3. Python Scripts Usage
If using the modular Python scripts:
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing script
python preprocessing_script.py

# Run model training
python model_training.py
```

## Dataset Information

### Source
- Dataset: Telco Customer Churn
- Platform: Kaggle
- Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Size: 7,043 customers, 21 features

### Key Features
- Demographic: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Account: tenure, Contract, PaperlessBilling, PaymentMethod
- Charges: MonthlyCharges, TotalCharges
- Target: Churn (Yes/No)

### Data Characteristics
- Missing Values: 11 entries in TotalCharges (handled)
- Class Distribution: 26.6% churn (imbalanced)
- Data Types: Mixed categorical and numerical

## Models Implemented

### 1. Decision Tree Models
- Baseline: max_depth=5, criterion='gini'
- Manual Tuning: Tested 16 parameter combinations manually
- Best Tuned: max_depth=10, criterion='entropy', min_samples_split=2, min_samples_leaf=1
- Evaluation: AUC=0.8349, Precision=0.6364, Recall=0.5053

### 2. Neural Network Models
- Baseline: 32-16-1 architecture, 20 epochs
- Manual Tuning: 256-128-64-1 with dropout, 50 epochs, early stopping
- Best Tuned: AUC=0.8408, Precision=0.6485, Recall=0.5080
- Training: Class weighting for imbalance, Adam optimizer (lr=0.001)

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Decision Tree (Baseline) | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8297 |
| Decision Tree (Tuned) | 0.7921 | 0.6364 | 0.5053 | 0.5633 | 0.8349 |
| Neural Network (Baseline) | 0.7885 | 0.6188 | 0.5294 | 0.5706 | 0.8340 |
| Neural Network (Tuned) | 0.7963 | 0.6485 | 0.5080 | 0.5697 | 0.8408 |

Best Overall Model: Neural Network (Tuned) - Highest AUC and Precision

## Ethical Considerations

### Fairness and Bias
- Analysis Conducted: Performance evaluation across gender and age groups
- Findings: Comparable performance across gender, some bias for senior citizens
- Mitigation: Class weighting, fairness-aware thresholds, regular audits

### Transparency
- Decision Trees: Clear decision rules and feature importance
- Neural Networks: SHAP explainability for individual predictions
- Documentation: Complete model cards with assumptions and limitations

### Privacy and Compliance
- Data Protection: PII removal (customerID), GDPR considerations
- Security: Secure model storage, access controls
- Accountability: Audit trails, human oversight for high-risk predictions

## Deployment Strategy

### Monitoring Framework
- Daily Tracking: Accuracy, precision, recall, F1, AUC
- Data Drift: Statistical tests for feature distribution changes
- Alert System: Automated notifications for performance degradation

### Continuous Improvement
- Weekly: Model updates with new customer data
- Monthly: Full retraining with hyperparameter review
- Quarterly: Comprehensive model and fairness audits

### Technical Infrastructure
- Containerization: Docker for consistent environments
- CI/CD: Automated testing and deployment pipeline
- Model Registry: Version control for models and data

### Business Integration
- CRM Integration: Automated high-risk customer identification
- Retention Workflows: Tailored intervention strategies
- Success Metrics: Retention rate improvement, ROI calculation

## Key Findings

### 1. Most Predictive Features
1. Contract Type: Month-to-month customers churn 43% vs Two-year 3%
2. Tenure: Strong negative correlation with churn (-0.35)
3. Internet Service: Fiber optic customers churn 42% vs DSL 19%
4. Payment Method: Electronic check 45% churn vs Bank transfer 16%
5. Technical Support: Without support 42% churn vs With support 15%

### 2. Model Performance Insights
- Best AUC: Neural Network Tuned (0.8408)
- Best Precision: Neural Network Tuned (0.6485) - Fewest false positives
- Best Recall: Decision Tree Baseline (0.5668) - Captures most churners
- Best Trade-off: Neural Network Tuned balances precision and recall

### 3. Business Recommendations
- High-Risk Groups: Focus on month-to-month contracts, fiber optic users, electronic check payers
- Intervention Timing: Early intervention for customers with tenure < 12 months
- Retention Strategies: Contract incentives, tech support bundling, payment method changes

## Limitations and Future Work

### Current Limitations
1. Dataset Size: 7,043 samples may limit complex model performance
2. Feature Scope: Limited behavioral and temporal data
3. Model Complexity: Neural network interpretability challenges
4. Real-world Dynamics: Static snapshot without market dynamics

### Future Enhancements
1. Advanced Models: Gradient boosting, ensemble methods, deep learning
2. Additional Data: Usage patterns, customer feedback, competitive data
3. Real-time Systems: Streaming data integration, online learning
4. Explainability: Advanced interpretation techniques, counterfactual analysis

## Contribution

### Author
- Name: [Amika Alankara ]
- Course: CM2604 Machine Learning
- Institution: [IIT]
- Academic Year: 2025/2026

### Acknowledgments
- Dataset provided by Kaggle and IBM
- Libraries: scikit-learn, TensorFlow, pandas, matplotlib, seaborn
- Course instructors and teaching assistants

### Academic Integrity
This work represents original implementation meeting all academic integrity requirements. All external sources are properly cited and referenced.

## License
This project is for academic purposes only. The dataset is publicly available on Kaggle under its own terms. Code is provided for educational use in accordance with university guidelines.

## Contact and Support

### For Technical Issues
1. Check the GitHub repository for latest updates
2. Review the troubleshooting section in the notebook
3. Contact via university email for coursework-specific queries

### Dataset Issues
- Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Ensure filename: WA_Fn-UseC_-Telco-Customer-Churn.csv
- Place in data/ folder before running

### Dependency Problems
```bash
# Create fresh environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
pip install -r requirements.txt
```

## Final Notes
- All project files are now properly organized and documented
- Two notebook versions provide flexibility for different use cases
- Python scripts enable modular and reusable code
- Repository is ready for submission and evaluation

Last Updated: 11 December 2025  
