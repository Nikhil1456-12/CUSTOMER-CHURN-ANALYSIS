# Customer Churn Analysis Dashboard (Enterprise Attrition Portal)

A professional, Business Intelligence (BI)-style web-based Customer Churn Analysis Dashboard that diagnoses churn indicators, evaluates model metrics, and predicts customer flight risks. This application is designed to help customer success teams and operation executives proactively identify at-risk subscribers and deploy targeted retention strategies.

## Deployed Streamlit Application
The application is deployed online and accessible at:
👉 **[https://customer-churn-analysis.streamlit.app](https://customer-churn-analysis.streamlit.app)**

## Project Objectives
1. **Executive Overview**: Provide real-time KPIs (Total Customers, Active Customers, Churn Rate %, Churned Count).
2. **Exploratory Data Analysis (EDA)**: Display 6 interactive, responsive visualizations explaining customer behavior and attrition triggers.
3. **Flight Risk Prediction**: Calculate individual churn probability and status using an inputs-driven machine learning model.
4. **Strategic Insights**: Highlight 5 core data-driven takeaways of customer behavior.
5. **Operational Recommendations**: Offer 5 concrete business strategies to reduce subscriber attrition.

## Technology Stack
- **Dashboard Interface**: Streamlit (Python)
- **Data Wrangling**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest Classifier)
- **Data Visualizations**: Plotly Express (Interactive), Matplotlib, Seaborn
- **Model Serialization**: Joblib
- **Report Generation**: ReportLab (Programmatic PDF Compiler)

## Repository Structure
```text
Customer-Churn-Analysis/
│
├── data/
│   └── Telco-Customer-Churn.csv    # Benchmark subscriber dataset (7,043 rows)
│
├── report/
│   └── CUSTOMER_CHURN_ANALYSIS.pdf # Pre-existing report copy
│
├── app.py                         # Main Streamlit web application
├── Customer_Churn_Analysis.ipynb  # Jupyter Notebook for EDA & Model Development
├── train_model.py                 # Reproducible model training & pipeline script
├── generate_pdf.py                # Programmatic PDF report builder
├── trained_model.pkl              # Serialized Random Forest model pipeline
├── requirements.txt               # Package dependencies list
├── Project_Report.pdf             # Generated corporate-style PDF project report
└── README.md                      # Documentation file
```

## Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-analysis.git
cd customer-churn-analysis
```

### 2. Set Up a Virtual Environment & Dependencies
We recommend using a virtual environment (venv) or conda:
```bash
# Using Python venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. (Optional) Run the Model Training Script
If you wish to re-train the Random Forest Classifier pipeline:
```bash
python train_model.py
```
This script will preprocess the Telco dataset, train a Random Forest model, evaluate it on an independent test split, print the performance classification report, and export a serialized pipeline to `trained_model.pkl`.

### 4. Run the Streamlit Application
Start the local web server:
```bash
streamlit run app.py
```
The dashboard will launch in your default browser at `http://localhost:8501`.

---

## Machine Learning Model Performance
To identify churned customers (the minority class of 26.5%), we trained a Random Forest Classifier with class weights. Since missing an at-risk customer (False Negative) is much more expensive than giving a retention incentive to a stable customer (False Positive), we optimized the tree parameters for **Recall**.

### Performance Summary (Stratified Test Set)
- **Overall Accuracy**: **75.80%**
- **Recall (Churn Class)**: **78.34%** (Correctly flags 78.3% of customers who will churn)
- **Precision (Churn Class)**: **52.98%**
- **ROC-AUC Score**: **84.23%** (Excellent discrimination threshold)

### Confusion Matrix Heatmap
```text
               Predicted Stay (0)   Predicted Churn (1)
Actual Stay         775                  260 (False Positive)
Actual Churn         81 (False Negative) 293 (True Positive)
```

---

## Executive Business Insights
1. **Contract Structure Vulnerability**: Month-to-month contracts have the highest risk. Over 42% of month-to-month subscribers churned, compared to less than 3% of two-year contract customers.
2. **Early-Lifecycle Risk**: The first 12 months are the most volatile. Median tenure for churned subscribers is only 10 months, compared to 38 months for active customers.
3. **Monthly Billing Sensitivity**: Higher rates spark attrition. The median monthly fee of churned customers ($79.68) is significantly higher than that of retained customers ($64.43).
4. **Electronic Check Friction**: Customers paying via Electronic Check show a 45% churn rate. Manual payment actions represent monthly friction checkpoints.
5. **Infrastructure Issues**: Fiber optic users churn at 41.8% compared to DSL users (18.9%), indicating quality expectations or pricing misalignment.

---

## Strategic Business Recommendations
1. **Incentivize Contract Migrations**: Offer a 10% discount on monthly plans for month-to-month users who sign up for annual commitments to lock in long-term Customer Lifetime Value (LTV).
2. **First-Year Onboarding Sequence**: Launch automated service check-ins at month 1, 3, and 6 to proactively intercept issues during the high-volatility period.
3. **Proactive Down-Sells**: Automatically offer value-optimized bundles to price-sensitive customers whose monthly charges cross $75, keeping them within the service ecosystem.
4. **Auto-Pay Credits**: Provide a one-time $5 billing credit for users who enroll in Automatic Credit Card or Bank Auto-Draft, removing payment decision friction.
5. **Quality Auditing for High-Value Tiers**: Perform automated connectivity audits for fiber optic lines to ensure premium paying customers are getting value, thereby reducing technical churn.

---

## Deployment to Streamlit Cloud (Preferred)
To make your application publicly accessible online:
1. Push this repository to GitHub.
2. Visit **[Streamlit Community Cloud](https://share.streamlit.io/)** and sign in using your GitHub credentials.
3. Click **New App**, select your repository (`CUSTOMER-CHURN-ANALYSIS`), branch (`main`), and path to the entry file (`app.py`).
4. Click **Deploy**. Your app will compile and become public at a URL like `https://customer-churn-analysis.streamlit.app`.
