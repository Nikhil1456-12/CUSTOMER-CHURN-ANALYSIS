import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ReportLab Imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Define cohesive corporate palette (Deep Indigo, Cool Gray, Teal, Coral)
PRIMARY_COLOR = colors.HexColor("#1A2B4C")    # Deep Navy/Indigo
SECONDARY_COLOR = colors.HexColor("#2A9D8F")  # Teal
ACCENT_COLOR = colors.HexColor("#E76F51")     # Coral
NEUTRAL_DARK = colors.HexColor("#2C3E50")     # Charcoal text
NEUTRAL_LIGHT = colors.HexColor("#F8F9FA")    # Off-white background
BORDER_COLOR = colors.HexColor("#BDC3C7")     # Light gray borders

# Canvas class to implement running headers/footers with 'Page X of Y'
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Suppress headers and footers on the cover page (Page 1)
        if self._pageNumber > 1:
            # Running Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(PRIMARY_COLOR)
            self.drawString(54, 750, "CUSTOMER CHURN ANALYSIS & RETENTION REPORT")
            self.setStrokeColor(PRIMARY_COLOR)
            self.setLineWidth(0.5)
            self.line(54, 742, letter[0] - 54, 742)
            
            # Running Footer
            self.line(54, 54, letter[0] - 54, 54)
            self.setFont("Helvetica", 8)
            self.setFillColor(NEUTRAL_DARK)
            self.drawString(54, 40, "Confidential - Business Intelligence Division")
            self.drawRightString(letter[0] - 54, 40, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def generate_visualizations(base_dir, pipeline):
    print("Generating report visualizations...")
    # Set style for report charts
    sns.set_theme(style="white")
    
    # 1. Generate Confusion Matrix Plot
    plt.figure(figsize=(5.5, 4.5))
    # Test set numbers from train_model.py: TP=293, FN=81, FP=260, TN=775
    # Actual confusion matrix: [[775, 260], [81, 293]]
    cm = np.array([[775, 260], [81, 293]])
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.color_palette("Blues", as_cmap=True), 
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.ylabel('Actual Churn Status', fontsize=11, fontweight='bold', labelpad=10)
    plt.xlabel('Predicted Churn Status', fontsize=11, fontweight='bold', labelpad=10)
    plt.xticks([0.5, 1.5], ['Stay (0)', 'Churn (1)'], fontsize=10)
    plt.yticks([0.5, 1.5], ['Stay (0)', 'Churn (1)'], fontsize=10, rotation=0)
    plt.title('Model Confusion Matrix Heatmap', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    cm_path = os.path.join(base_dir, 'temp_cm.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Generate Feature Importance Plot
    # Extract features and importances
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'PaymentMethod']
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'InternetService', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cats = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_features = encoded_cats + numerical_cols
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    feature_imp_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    agg_importances = {}
    for col in features:
        if col in categorical_cols:
            agg_importances[col] = feature_imp_df[feature_imp_df['feature'].str.startswith(col)]['importance'].sum()
        else:
            agg_importances[col] = feature_imp_df[feature_imp_df['feature'] == col]['importance'].values[0]
            
    agg_imp_df = pd.DataFrame(list(agg_importances.items()), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)
    
    # Rename for readability
    rename_dict = {
        'Contract': 'Contract Type',
        'tenure': 'Customer Tenure (Months)',
        'MonthlyCharges': 'Monthly Charges ($)',
        'TotalCharges': 'Total Charges ($)',
        'InternetService': 'Internet Service Type',
        'PaymentMethod': 'Payment Method',
        'Dependents': 'Dependents Status',
        'gender': 'Gender',
        'Partner': 'Partner Status',
        'SeniorCitizen': 'Senior Citizen Status',
        'PhoneService': 'Phone Service Status'
    }
    agg_imp_df['feature'] = agg_imp_df['feature'].map(rename_dict)
    
    plt.figure(figsize=(6.5, 4))
    colors_list = sns.color_palette("viridis", len(agg_imp_df))
    sns.barplot(data=agg_imp_df.head(10), x='importance', y='feature', palette=colors_list)
    plt.title('Top 10 Feature Importances (Aggregated)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Aggregated Gini Importance Score', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('')
    plt.tight_layout()
    fi_path = os.path.join(base_dir, 'temp_fi.png')
    plt.savefig(fi_path, dpi=300)
    plt.close()
    
    return cm_path, fi_path, agg_imp_df

def build_pdf_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'trained_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train_model.py first.")
        
    pipeline = joblib.load(model_path)
    
    # Generate images
    cm_path, fi_path, agg_imp_df = generate_visualizations(base_dir, pipeline)
    
    pdf_path = os.path.join(base_dir, 'Project_Report.pdf')
    print(f"Compiling PDF report to {pdf_path}...")
    
    # Page Margins (0.75 in or 54 points)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    # Set styles
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=32,
        leading=38,
        textColor=PRIMARY_COLOR,
        alignment=0, # Left aligned
        spaceAfter=15
    ))
    
    styles.add(ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=16,
        leading=22,
        textColor=SECONDARY_COLOR,
        alignment=0,
        spaceAfter=40
    ))
    
    styles.add(ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=NEUTRAL_DARK,
        alignment=0
    ))
    
    styles.add(ParagraphStyle(
        'ReportHeading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY_COLOR,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        'ReportHeading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=SECONDARY_COLOR,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=NEUTRAL_DARK,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        'ReportBodyBold',
        parent=styles['ReportBody'],
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        'InsightBoxText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13,
        textColor=PRIMARY_COLOR
    ))
    
    story = []
    
    # ------------------ COVER PAGE ------------------
    story.append(Spacer(1, 1.2 * inch))
    # Title Block
    story.append(Paragraph("Customer Churn Analysis<br/>Using Machine Learning", styles['CoverTitle']))
    # Decorative horizontal line
    t_line = Table([[""]], colWidths=[letter[0] - 108], rowHeights=[4])
    t_line.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), SECONDARY_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_line)
    story.append(Spacer(1, 20))
    story.append(Paragraph("A Business Intelligence Report on Subscriber Behavior and Predictive Retention Strategies", styles['CoverSubtitle']))
    
    story.append(Spacer(1, 1.8 * inch))
    
    # Metadata Block
    date_str = datetime.now().strftime("%B %d, %Y")
    meta_text = f"""
    <b>Prepared For:</b> Executive Operations & Customer Success Teams<br/>
    <b>Author:</b> Lead Data Scientist / Churn Analysis Dashboard<br/>
    <b>Model Version:</b> Random Forest Classifier (v1.0.0)<br/>
    <b>Date:</b> {date_str}<br/>
    <b>Dataset Source:</b> Telco Customer Retention Repository
    """
    story.append(Paragraph(meta_text, styles['CoverMeta']))
    story.append(PageBreak())
    
    # ------------------ SECTION 1: EXECUTIVE SUMMARY ------------------
    story.append(Paragraph("Executive Summary", styles['ReportHeading1']))
    story.append(Paragraph(
        "Customer retention is the primary driver of profitability in modern subscription-based services. "
        "Acquiring new customers is mathematically and operationally more expensive than retaining existing ones. "
        "This report presents a comprehensive data-driven analysis of subscriber behavior, identifies "
        "key indicators of attrition, and evaluates a predictive Machine Learning model designed to flag at-risk subscribers. "
        "By leveraging a Random Forest model trained on demographic, account, and transactional characteristics, "
        "we achieve a robust predictive capability (78% recall on churned customers), allowing "
        "operations teams to proactively intervene and safeguard recurring revenue.",
        styles['ReportBody']
    ))
    
    # Quick Facts Table
    summary_data = [
        [Paragraph("<b>Key Metric</b>", styles['ReportBodyBold']), Paragraph("<b>Value</b>", styles['ReportBodyBold']), Paragraph("<b>Business Operational Meaning</b>", styles['ReportBodyBold'])],
        ["Total Customers Evaluated", "7,043", "Complete baseline customer base analyzed for patterns."],
        ["Active Customer Base", "5,174 (73.5%)", "Baseline cohort that remained active and loyal."],
        ["Churned Customers", "1,869 (26.5%)", "Subscribers lost during the observation window."],
        ["Target Retention Accuracy", "75.8%", "Model's capability to generalize across all customer groups."],
        ["Retention Recall Rate", "78.3%", "Proportion of actually churned customers correctly flagged by the model."]
    ]
    t_summary = Table(summary_data, colWidths=[2.0*inch, 1.2*inch, 3.8*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), NEUTRAL_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(Spacer(1, 10))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # ------------------ SECTION 2: DATASET & PREPROCESSING ------------------
    story.append(Paragraph("Dataset & Feature Preprocessing", styles['ReportHeading1']))
    story.append(Paragraph(
        "The analysis is conducted on the standard Telco Customer Churn dataset, comprising 7,043 customer records "
        "and 21 variables. Before modeling, the data undergoes rigorous preprocessing to ensure mathematical compatibility "
        "with our Random Forest pipeline:",
        styles['ReportBody']
    ))
    story.append(Paragraph(
        "<b>Data Cleaning:</b> The `TotalCharges` column contained 11 blank spaces, representing new customers with 0 months of "
        "tenure. These were identified, filled with a baseline of $0.00, and converted from text string to float64 numeric type. "
        "This ensures that new subscribers are not discarded from analysis.<br/>"
        "<b>Categorical Encoding:</b> Variables like `Contract`, `InternetService`, and `PaymentMethod` were transformed using "
        "One-Hot Encoding. This converts categorical labels into binary dummy features, which the tree splits can evaluate.<br/>"
        "<b>Class Balancing:</b> The dataset exhibits moderate class imbalance (26.5% churned vs 73.5% retained). To address this, "
        "the model training pipeline utilizes stratified train-test splits and implements class weight penalization, forcing the tree "
        "leaves to place higher cost on misclassifying the minority churn class. This increases model sensitivity (Recall).",
        styles['ReportBody']
    ))
    story.append(PageBreak())
    
    # ------------------ SECTION 3: EXPLORATORY DATA ANALYSIS (EDA) ------------------
    story.append(Paragraph("Exploratory Data Analysis (EDA) Insights", styles['ReportHeading1']))
    story.append(Paragraph(
        "Before feeding features into the machine learning algorithm, we performed a multi-dimensional analysis to isolate "
        "the key variables that correlate with churn. Five critical patterns emerged:",
        styles['ReportBody']
    ))
    
    eda_insights = [
        ("1. Contract Type Impact", "Contract type is the single most predictive feature. Customers on <b>Month-to-month contracts</b> exhibit a massive churn rate compared to those on 1-year or 2-year commitments. Month-to-month subscribers represent the 'highest-risk' group because their switching cost is zero."),
        ("2. Tenure & Customer Lifecycle", "Customers who have been with the company for less than 12 months show the highest propensity to churn. Once a customer passes the 24-month mark, their likelihood of leaving drops dramatically, indicating that customer loyalty builds over time."),
        ("3. High Cost Sensitivities", "Box plot analysis reveals that churned customers have a significantly higher median <b>Monthly Charge ($79.68)</b> than active customers ($64.43). Higher billing amounts directly provoke customer attrition, highlighting the need for competitive tier pricing."),
        ("4. Payment Method & Digital Friction", "Subscribers paying via <b>Electronic Check</b> churn at a substantially higher rate than those using automatic payment methods (Credit Card, Bank Transfer) or Mailed Checks. Electronic check payments require monthly manual action, presenting a recurring monthly friction point that triggers exit decisions."),
        ("5. Internet Infrastructure", "Subscribers on <b>Fiber Optic</b> connections show elevated churn compared to DSL users. While Fiber Optic offers higher speeds, it is also higher cost, and users may suffer from service delivery issues or expectations mismatches.")
    ]
    
    for title, desc in eda_insights:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", styles['ReportBody']))
        story.append(Spacer(1, 4))
        
    story.append(Spacer(1, 10))
    
    # ------------------ SECTION 4: MACHINE LEARNING MODEL ------------------
    story.append(Paragraph("Machine Learning Model Construction", styles['ReportHeading1']))
    story.append(Paragraph(
        "A <b>Random Forest Classifier</b> was trained on 80% of the customer database. Random Forest is an ensemble method "
        "that aggregates predictions from a collection of decision trees. It is highly robust, handles non-linear interactions "
        "between variables automatically, and is less prone to overfitting than single decision trees.",
        styles['ReportBody']
    ))
    story.append(Paragraph(
        "The classification pipeline is built using Scikit-Learn's `Pipeline` and `ColumnTransformer` frameworks. "
        "This architectural pattern ensures that all preprocessing rules are tightly packaged with the estimator, eliminating "
        "data leakage and making deployment into production (`app.py`) simple and bulletproof. The final model uses "
        "150 estimators, a max depth of 8 to prevent overfitting, and a minimum of 4 samples per leaf to ensure robust leaf statistics.",
        styles['ReportBody']
    ))
    
    story.append(PageBreak())
    
    # ------------------ SECTION 5: MODEL PERFORMANCE ------------------
    story.append(Paragraph("Model Performance & Evaluation", styles['ReportHeading1']))
    story.append(Paragraph(
        "The model is validated on an independent, stratified test set representing 20% of the original data (1,409 customers). "
        "Since the primary objective is to catch as many churned customers as possible, we prioritize the <b>Recall</b> (sensitivity) "
        "metric over raw accuracy alone.",
        styles['ReportBody']
    ))
    
    # Metrics Table
    metrics_data = [
        [Paragraph("<b>Evaluation Metric</b>", styles['ReportBodyBold']), Paragraph("<b>Score</b>", styles['ReportBodyBold']), Paragraph("<b>Business Value</b>", styles['ReportBodyBold'])],
        ["Accuracy", "75.80%", "Indicates that the model correctly predicts overall status 3 out of 4 times."],
        ["Recall (Churn class)", "78.34%", "We successfully capture 78.3% of all customers who will actually churn."],
        ["Precision (Churn class)", "52.98%", "Out of all predicted churners, 53% actually leave (the rest represent a margin for safe retention offers)."],
        ["F1-Score (Churn class)", "63.22%", "Balanced harmonic mean of precision and recall for the target group."],
        ["ROC-AUC Score", "84.23%", "High discriminative power between churners and non-churners across all thresholds."]
    ]
    t_metrics = Table(metrics_data, colWidths=[2.0*inch, 1.2*inch, 3.8*inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,1), (-1,-1), NEUTRAL_LIGHT),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))
    
    # Add Confusion Matrix image side-by-side with explanation
    img_cm = Image(cm_path, width=3.2*inch, height=2.6*inch)
    cm_desc = """
    <b>Understanding the Confusion Matrix:</b><br/><br/>
    - <b>True Negatives (TN = 775):</b> Model correctly predicted 775 active customers who stayed.<br/>
    - <b>True Positives (TP = 293):</b> Model correctly identified 293 customers who churned. These are high-value targets for proactive marketing offers.<br/>
    - <b>False Positives (FP = 260):</b> Model flagged 260 customers as churn risk who actually stayed. While technically errors, targeting these customers with loyalty rewards helps reinforce retention.<br/>
    - <b>False Negatives (FN = 81):</b> Model missed 81 customers who churned. This represents a 21.7% leak rate.
    """
    p_cm_desc = Paragraph(cm_desc, styles['ReportBody'])
    
    t_cm_block = Table([[img_cm, p_cm_desc]], colWidths=[3.4*inch, 3.6*inch])
    t_cm_block.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (1,0), (1,0), 10),
    ]))
    story.append(t_cm_block)
    story.append(PageBreak())
    
    # ------------------ SECTION 6: FEATURE IMPORTANCE ------------------
    story.append(Paragraph("Feature Importance Analysis", styles['ReportHeading1']))
    story.append(Paragraph(
        "By analyzing the Random Forest's Gini impurity reduction and aggregating the one-hot encoded variables back to their "
        "original categorical parents, we establish the absolute hierarchy of variables driving customer churn.",
        styles['ReportBody']
    ))
    
    # Add Feature Importance image side-by-side with table
    img_fi = Image(fi_path, width=3.6*inch, height=2.2*inch)
    
    # Table of importances
    fi_table_data = [[Paragraph("<b>Feature Name</b>", styles['ReportBodyBold']), Paragraph("<b>Aggregated Score</b>", styles['ReportBodyBold'])]]
    for idx, row in agg_imp_df.head(6).iterrows():
        fi_table_data.append([row['feature'], f"{row['importance']:.2%}"])
    
    t_fi_table = Table(fi_table_data, colWidths=[2.0*inch, 1.2*inch])
    t_fi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,1), (-1,-1), NEUTRAL_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ]))
    
    t_fi_block = Table([[img_fi, t_fi_table]], colWidths=[3.8*inch, 3.2*inch])
    t_fi_block.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (1,0), (1,0), 10),
    ]))
    story.append(t_fi_block)
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "The analysis demonstrates that <b>Contract Type (26.02%)</b> is the dominant factor, followed closely by "
        "<b>Tenure (16.61%)</b>, <b>Monthly Charges (13.99%)</b>, and <b>Total Charges (13.94%)</b>. "
        "Collectively, these top four features account for over <b>70%</b> of the predictive power of the model. "
        "This indicates that economic and commitment factors outweigh demographic characteristics in predicting customer decisions.",
        styles['ReportBody']
    ))
    story.append(Spacer(1, 10))
    
    # ------------------ SECTION 7 & 8: BUSINESS INSIGHTS & RECOMMENDATIONS ------------------
    story.append(Paragraph("Business Insights & Actionable Recommendations", styles['ReportHeading1']))
    story.append(Paragraph(
        "Based on our statistical analysis and the machine learning model, we have formulated the following strategic "
        "recommendations to reduce customer churn and protect recurring revenue:",
        styles['ReportBody']
    ))
    
    recs = [
        ("1. Incentivize Contract Transitions", "Since Month-to-month contracts represent the highest churn risk, the business must implement a targeted migration campaign. Offer a <b>10-15% discount</b> on monthly fees for customers who transition to a 1-year or 2-year contract. The loss in marginal fee is heavily offset by the guaranteed customer lifetime value (LTV)."),
        ("2. Mitigate Early-Stage Churn (Onboarding)", "The first 12 months are the most critical. Establish a dedicated customer onboarding flow, including automated check-ins at month 1, 3, and 6. Provide proactive tech support, user tutorials, and minor loyalty incentives (e.g., free premium channel trial) during this window to anchor the customer."),
        ("3. Value-Added Pricing Optimization", "High monthly charges trigger churn. Implement standard price-sensitivity thresholds. When a customer's monthly charge exceeds $75.00, proactively trigger account reviews. Offer 'down-sell' paths to cheaper, value-aligned service packages rather than letting the customer cancel entirely."),
        ("4. Encourage Automated Billing Enrollment", "Payment via Electronic Check represents manual monthly friction. Create a small credit incentive (e.g., a one-time $5 bill credit) for users who sign up for Automatic Credit Card or Bank Auto-Draft payments. Moving users to auto-pay removes recurring decision friction and drastically lowers churn."),
        ("5. Proactive High-Risk Marketing Retention Campaigns", "Use the Streamlit dashboard ML tool to run batch predictions on existing users weekly. Extract all customers with a churn probability score <b>exceeding 70%</b>. Route these leads to a high-priority customer success team authorized to deploy personalized retention packages.")
    ]
    
    for title, desc in recs:
        p_rec = Paragraph(f"<b>{title}:</b> {desc}", styles['ReportBody'])
        # Wrap each recommendation in a light grey box for professional styling
        t_rec_box = Table([[p_rec]], colWidths=[7.0*inch])
        t_rec_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), NEUTRAL_LIGHT),
            ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(t_rec_box)
        story.append(Spacer(1, 8))
        
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF Report generated successfully!")
    
    # Cleanup temporary images
    try:
        os.remove(cm_path)
        os.remove(fi_path)
        print("Temporary visualization files removed.")
    except Exception as e:
        print("Error removing temporary files:", e)

if __name__ == '__main__':
    build_pdf_report()
