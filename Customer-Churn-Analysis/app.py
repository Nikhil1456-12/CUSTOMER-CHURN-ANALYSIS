import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Page config
st.set_page_config(
    page_title="Customer Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern BI styling (Navy, Teal, Coral scheme)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom CSS KPI Cards */
    .kpi-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1A2B4C;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-card-title {
        font-size: 14px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-card-value {
        font-size: 32px;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Styled lists for Insights & Recommendations */
    .styled-list-item {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 4px solid #2A9D8F;
    }
    .styled-list-title {
        font-weight: 600;
        color: #1A2B4C;
        margin-bottom: 4px;
        font-size: 16px;
    }
    .styled-list-desc {
        color: #475569;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Insight Alert boxes */
    .insight-alert {
        background-color: #F0FDFA;
        border-left: 4px solid #2A9D8F;
        padding: 12px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 14px;
        color: #0F766E;
    }
</style>
""", unsafe_allow_html=True)

def map_columns(df):
    col_mapping = {}
    
    # 1. gender
    for col in df.columns:
        if col.lower() in ['gender', 'sex']:
            col_mapping[col] = 'gender'
            break
            
    # 2. tenure
    for col in df.columns:
        if col.lower() in ['tenure', 'months', 'tenure_months', 'customer_tenure']:
            col_mapping[col] = 'tenure'
            break
            
    # 3. Contract
    for col in df.columns:
        if col.lower() in ['contract', 'contract_length', 'contract length', 'contract_term', 'contract term']:
            col_mapping[col] = 'Contract'
            break
            
    # 4. InternetService
    for col in df.columns:
        if col.lower() in ['internetservice', 'internet service', 'internet_service', 'subscription_type', 'subscription type', 'service_type', 'service type']:
            col_mapping[col] = 'InternetService'
            break
            
    # 5. TotalCharges
    for col in df.columns:
        if col.lower() in ['totalcharges', 'total charges', 'total_charges', 'total spend', 'total_spend', 'totalspend']:
            col_mapping[col] = 'TotalCharges'
            break
            
    # 6. Churn
    for col in df.columns:
        if col.lower() in ['churn', 'churn_status', 'churn status', 'left']:
            col_mapping[col] = 'Churn'
            break

    # 7. PaymentMethod
    for col in df.columns:
        if col.lower() in ['paymentmethod', 'payment method', 'payment_method', 'payment delay', 'payment_delay']:
            col_mapping[col] = 'PaymentMethod'
            break

    # Rename columns using the mapping
    df = df.rename(columns=col_mapping)
    
    # Clean and convert numeric columns to proper float/int types early
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
        
    if 'TotalCharges' in df.columns:
        if df['TotalCharges'].dtype == object:
            df['TotalCharges'] = df['TotalCharges'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
        
    if 'MonthlyCharges' in df.columns:
        if df['MonthlyCharges'].dtype == object:
            df['MonthlyCharges'] = df['MonthlyCharges'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)

    # Check for Age to compute SeniorCitizen
    if 'SeniorCitizen' not in df.columns:
        for col in df.columns:
            if col.lower() == 'age':
                df['SeniorCitizen'] = (df[col] >= 65).astype(int)
                break
                
    # If MonthlyCharges is missing but we have TotalCharges and tenure
    if 'MonthlyCharges' not in df.columns and 'TotalCharges' in df.columns and 'tenure' in df.columns:
        # Avoid division by zero
        temp_tenure = df['tenure'].replace(0, 1)
        df['MonthlyCharges'] = df['TotalCharges'] / temp_tenure
        
    # Fill in missing required columns with default values if they are not in df
    required_cols = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'InternetService': 'DSL',
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.0,
        'TotalCharges': 0.0,
        'Churn': 'No'
    }
    
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default
            
    # Standardize values for Contract:
    if 'Contract' in df.columns:
        def standardize_contract(val):
            val_str = str(val).lower().strip()
            if 'month' in val_str or val_str == '1' or val_str == 'monthly' or '0' in val_str:
                return 'Month-to-month'
            elif 'one' in val_str or '12' in val_str or '1 year' in val_str or val_str == '12' or 'annual' in val_str:
                return 'One year'
            elif 'two' in val_str or '24' in val_str or '2 year' in val_str or val_str == '24' or 'biennial' in val_str:
                return 'Two year'
            else:
                return 'Month-to-month'
        df['Contract'] = df['Contract'].apply(standardize_contract)
        
    # Standardize values for Churn:
    if 'Churn' in df.columns:
        def standardize_churn(val):
            if isinstance(val, (int, float, np.number)):
                if val == 1 or val == 1.0:
                    return 'Yes'
                else:
                    return 'No'
            val_str = str(val).lower().strip()
            if val_str in ['yes', '1', '1.0', 'true', 'churned', 'y']:
                return 'Yes'
            else:
                return 'No'
        df['Churn'] = df['Churn'].apply(standardize_churn)
        
    # Standardize values for InternetService:
    if 'InternetService' in df.columns:
        def standardize_internet(val):
            val_str = str(val).lower().strip()
            if 'fiber' in val_str or 'optic' in val_str or 'premium' in val_str:
                return 'Fiber optic'
            elif 'dsl' in val_str or 'basic' in val_str or 'cable' in val_str:
                return 'DSL'
            elif 'no' in val_str or 'none' in val_str or 'false' in val_str:
                return 'No'
            else:
                return 'DSL'
        df['InternetService'] = df['InternetService'].apply(standardize_internet)

    # Standardize values for PaymentMethod:
    if 'PaymentMethod' in df.columns:
        def standardize_payment(val):
            val_str = str(val).lower().strip()
            if 'electronic' in val_str or 'check' in val_str:
                return 'Electronic check'
            elif 'mailed' in val_str or 'mail' in val_str:
                return 'Mailed check'
            elif 'bank' in val_str or 'transfer' in val_str:
                return 'Bank transfer (automatic)'
            elif 'credit' in val_str or 'card' in val_str:
                return 'Credit card (automatic)'
            else:
                return 'Electronic check'
        df['PaymentMethod'] = df['PaymentMethod'].apply(standardize_payment)
        
    return df

# Helper function to load and preprocess data
@st.cache_data
def load_data(file_path_or_buffer=None):
    if file_path_or_buffer is None:
        file_path_or_buffer = os.path.join(os.path.dirname(__file__), 'data', 'Telco-Customer-Churn.csv')
    
    df = pd.read_csv(file_path_or_buffer)
    
    # Map and fill missing columns dynamically
    df = map_columns(df)
    
    # Clean TotalCharges
    if 'TotalCharges' in df.columns:
        if df['TotalCharges'].dtype == object:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    return df

# Initialize session state for dataset
if 'df' not in st.session_state:
    try:
        st.session_state.df = load_data()
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        st.session_state.df = None

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3067/3067260.png", width=80)
st.sidebar.title("Churn Analytics")
st.sidebar.markdown("*Enterprise Customer Retention Suite*")
st.sidebar.markdown("---")

pages = [
    "🏠 Home Dashboard",
    "📁 Dataset Upload",
    "📊 Exploratory Data Analysis (EDA)",
    "🔮 Machine Learning Prediction",
    "📈 Model Performance",
    "🧬 Feature Importance",
    "💡 Business Insights",
    "🎯 Recommendations"
]
selected_page = st.sidebar.radio("Navigation Menu", pages)

st.sidebar.markdown("---")
# Download PDF report button
try:
    with open("Project_Report.pdf", "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    st.sidebar.download_button(
        label="📥 Download PDF Project Report",
        data=pdf_bytes,
        file_name="Project_Report.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.sidebar.warning("Generate PDF to enable report download.")

# Main title header
st.markdown("<h1 style='color: #1A2B4C; font-weight: 700;'>Customer Churn Analysis Web Application</h1>", unsafe_allow_html=True)
st.markdown("---")

df = st.session_state.df

# Page 1: Home Dashboard
if selected_page == "🏠 Home Dashboard":
    st.markdown("### Executive Performance Dashboard")
    st.write(
        "Welcome to the Customer Churn Analysis Portal. This enterprise platform leverages machine learning "
        "to diagnose customer churn, predict individual customer flight risks, and surface strategic retention pathways."
    )
    
    if df is not None:
        # Compute KPIs
        total_customers = len(df)
        
        # Determine churn mapping
        if 'Churn' in df.columns:
            # Check if churn is text (Yes/No) or numeric (1/0)
            if df['Churn'].dtype == object:
                churned_count = (df['Churn'].str.lower() == 'yes').sum()
            else:
                churned_count = (df['Churn'] == 1).sum()
        else:
            churned_count = 0
            
        active_count = total_customers - churned_count
        churn_rate = (churned_count / total_customers) * 100 if total_customers > 0 else 0
        
        # Display KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #1A2B4C;">
                <div class="kpi-card-title">Total Customers</div>
                <div class="kpi-card-value">{total_customers:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #2A9D8F;">
                <div class="kpi-card-title">Active Customers</div>
                <div class="kpi-card-value">{active_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #E76F51;">
                <div class="kpi-card-title">Churned Customers</div>
                <div class="kpi-card-value">{churned_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: #E9C46A;">
                <div class="kpi-card-title">Churn Rate</div>
                <div class="kpi-card-value">{churn_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Add summary chart of Churn and Contract type side by side
        col1, col2 = st.columns(2)
        with col1:
            fig_churn = px.pie(
                df, names='Churn', title="Overall Retention Ratio",
                color_discrete_sequence=['#2A9D8F', '#E76F51'], hole=0.5
            )
            fig_churn.update_layout(title=dict(text="<b>Overall Retention Ratio</b>", font=dict(size=16, family="Inter", color="#1A2B4C")))
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with col2:
            fig_contract = px.histogram(
                df, x='Contract', color='Churn', barmode='group',
                title="Customer Status by Contract Type",
                color_discrete_sequence=['#2A9D8F', '#E76F51'],
                labels={'Contract': 'Contract Term'}
            )
            fig_contract.update_layout(title=dict(text="<b>Customer Status by Contract Type</b>", font=dict(size=16, family="Inter", color="#1A2B4C")))
            st.plotly_chart(fig_contract, use_container_width=True)
    else:
        st.warning("Please upload a dataset to begin the analysis.")

# Page 2: Dataset Upload
elif selected_page == "📁 Dataset Upload":
    st.markdown("### Dataset Upload & Preview")
    st.write(
        "Upload your business's customer dataset (CSV format) to perform churn profiling and predict flight probabilities. "
        "The system will automatically fall back to the standard Telco Customer database if no file is provided."
    )
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if uploaded_file is not None:
        try:
            st.session_state.df = load_data(uploaded_file)
            st.success("Dataset uploaded and processed successfully!")
            # Re-read session df
            df = st.session_state.df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            
    if df is not None:
        st.markdown("#### Dataset Specifications")
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.write(f"**Number of Rows:** {df.shape[0]}")
        with spec_col2:
            st.write(f"**Number of Columns:** {df.shape[1]}")
            
        st.markdown("#### Sample Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Upload a dataset to see details.")

# Page 3: Exploratory Data Analysis (EDA)
elif selected_page == "📊 Exploratory Data Analysis (EDA)":
    st.markdown("### Customer Demographics & Behavioral Profiling")
    st.write(
        "Examine key metrics and relationships in the customer base. Use the interactive Plotly graphs "
        "to filter, zoom, and inspect subsets of subscribers."
    )
    
    if df is not None:
        # Standardize target Churn column text format
        df_eda = df.copy()
        if 'Churn' in df_eda.columns:
            if df_eda['Churn'].dtype == object:
                df_eda['Churn_Display'] = df_eda['Churn'].map({'Yes': 'Churned', 'No': 'Retained'})
            else:
                df_eda['Churn_Display'] = df_eda['Churn'].map({1: 'Churned', 0: 'Retained'})
        else:
            st.error("No 'Churn' column found in dataset. EDA charts require a target 'Churn' variable.")
            st.stop()
            
        col1, col2 = st.columns(2)
        
        # Viz 1: Churn Distribution
        with col1:
            st.markdown("#### 1. Customer Churn Distribution")
            fig1 = px.pie(
                df_eda, names='Churn_Display', hole=0.5,
                color='Churn_Display',
                color_discrete_map={'Retained': '#2A9D8F', 'Churned': '#E76F51'}
            )
            fig1.update_layout(legend_title="Customer Status", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(
                '<div class="insight-alert"><b>Insight:</b> The baseline dataset exhibits a 26.5% churn rate. '
                'Over a quarter of the customer base left during the observation period.</div>',
                unsafe_allow_html=True
            )
            
        # Viz 2: Contract Type vs Churn
        with col2:
            st.markdown("#### 2. Contract Type vs Churn")
            fig2 = px.histogram(
                df_eda, x='Contract', color='Churn_Display', barmode='group',
                color_discrete_map={'Retained': '#2A9D8F', 'Churned': '#E76F51'},
                labels={'Contract': 'Contract Type'}
            )
            fig2.update_layout(xaxis_title="Contract Type", yaxis_title="Customer Count", legend_title="Status")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                '<div class="insight-alert"><b>Insight:</b> Month-to-month contracts account for the overwhelming majority '
                'of churned customers. Long-term contract commitments (1 or 2 years) show very high retention.</div>',
                unsafe_allow_html=True
            )
            
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        # Viz 3: Tenure vs Churn
        with col3:
            st.markdown("#### 3. Tenure vs Churn")
            fig3 = px.box(
                df_eda, x='Churn_Display', y='tenure',
                color='Churn_Display',
                color_discrete_map={'Retained': '#2A9D8F', 'Churned': '#E76F51'},
                labels={'tenure': 'Tenure (Months)', 'Churn_Display': 'Status'}
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                '<div class="insight-alert"><b>Insight:</b> Customers with short tenure are highly prone to churn. '
                'The median tenure for churned customers is around 10 months, compared to 38 months for active customers.</div>',
                unsafe_allow_html=True
            )
            
        # Viz 4: Monthly Charges vs Churn
        with col4:
            st.markdown("#### 4. Monthly Charges vs Churn")
            fig4 = px.box(
                df_eda, x='Churn_Display', y='MonthlyCharges',
                color='Churn_Display',
                color_discrete_map={'Retained': '#2A9D8F', 'Churned': '#E76F51'},
                labels={'MonthlyCharges': 'Monthly Charges ($)', 'Churn_Display': 'Status'}
            )
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(
                '<div class="insight-alert"><b>Insight:</b> Churned customers pay significantly higher monthly fees '
                '(median: $79.68) than retained subscribers (median: $64.43). Pricing pressure stimulates exit.</div>',
                unsafe_allow_html=True
            )
            
        st.markdown("---")
        col5, col6 = st.columns(2)
        
        # Viz 5: Payment Method vs Churn
        with col5:
            st.markdown("#### 5. Payment Method vs Churn")
            fig5 = px.histogram(
                df_eda, y='PaymentMethod', color='Churn_Display', barmode='group',
                color_discrete_map={'Retained': '#2A9D8F', 'Churned': '#E76F51'},
                labels={'PaymentMethod': 'Payment Method'}
            )
            fig5.update_layout(yaxis_title="Payment Method", xaxis_title="Customer Count", legend_title="Status")
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown(
                '<div class="insight-alert"><b>Insight:</b> Customers paying with Electronic Check show disproportionately '
                'high churn. Moving users to automated bank transfer or credit card auto-pay reduces friction.</div>',
                unsafe_allow_html=True
            )
            
        # Viz 6: Correlation Heatmap
        with col6:
            st.markdown("#### 6. Numerical Feature Correlations")
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
            if 'Churn' in df_eda.columns:
                # Map Churn to numeric for correlation mapping
                if df_eda['Churn'].dtype == object:
                    df_eda['Churn_Numeric'] = df_eda['Churn'].map({'Yes': 1, 'No': 0})
                else:
                    df_eda['Churn_Numeric'] = df_eda['Churn']
                numeric_cols.append('Churn_Numeric')
            
            corr_df = df_eda[numeric_cols].corr()
            
            fig6 = px.imshow(
                corr_df, text_auto=".2f",
                color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation")
            )
            fig6.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown(
                '<div class="insight-alert"><b>Insight:</b> Tenure has a strong negative correlation with churn (-0.35), '
                'while Monthly Charges has a positive correlation (+0.19). Higher prices and short tenures are primary warning signals.</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Please upload a dataset to begin the analysis.")

# Page 4: Machine Learning Prediction
elif selected_page == "🔮 Machine Learning Prediction":
    st.markdown("### Predict Customer Flight Risk")
    st.write(
        "Enter customer subscription characteristics below to calculate the mathematical probability "
        "of that customer churn event."
    )
    
    # Load model pipeline
    model_path = "trained_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Trained model pipeline file (`{model_path}`) is missing! Please train the model first.")
    else:
        pipeline = joblib.load(model_path)
        
        st.markdown("#### Customer Account Information Form")
        
        # Build 3 column layout for form inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen Status", ["No (0)", "Yes (1)"])
            partner = st.selectbox("Partner Status", ["Yes", "No"])
            dependents = st.selectbox("Dependents Status", ["Yes", "No"])
            
        with col2:
            tenure = st.slider("Customer Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
            phone_service = st.selectbox("Phone Service Enabled", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            
        with col3:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=10.0)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        predict_btn = st.button("Predict Churn Risk Status", type="primary", use_container_width=True)
        
        if predict_btn:
            # Map input senior citizen to binary
            sc_val = 1 if "Yes" in senior_citizen else 0
            
            # Create a dataframe for the input
            input_data = pd.DataFrame([{
                'gender': gender,
                'SeniorCitizen': sc_val,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'Contract': contract,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }])
            
            with st.spinner("Analyzing risk metrics..."):
                # Run prediction
                prediction = pipeline.predict(input_data)[0]
                proba = pipeline.predict_proba(input_data)[0][1]
                
            st.markdown("---")
            st.markdown("### Prediction Outcome")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.markdown("""
                    <div style="background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 24px; border-radius: 8px;">
                        <h4 style="color: #991B1B; margin-top: 0;">⚠️ HIGH FLIGHT RISK</h4>
                        <p style="color: #7F1D1D; font-size: 16px; font-weight: 600;">Customer is predicted to CHURN.</p>
                        <p style="color: #991B1B; font-size: 14px;">Immediate proactive retention action is recommended. Review contract tier migrations or payment automations.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 24px; border-radius: 8px;">
                        <h4 style="color: #065F46; margin-top: 0;">✅ STABLE SUBSCRIBER</h4>
                        <p style="color: #064E3B; font-size: 16px; font-weight: 600;">Customer is predicted to STAY.</p>
                        <p style="color: #065F46; font-size: 14px;">Customer shows standard loyalty signals. Regular promotional campaigns and customer service check-ins should continue.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with res_col2:
                # Draw a gauge chart for flight risk
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability (%)", 'font': {'size': 18, 'color': '#1A2B4C', 'family': 'Inter'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#E76F51" if proba > 0.5 else "#2A9D8F"},
                        'bgcolor': "#E2E8F0",
                        'borderwidth': 2,
                        'bordercolor': "#F8FAFC",
                        'steps': [
                            {'range': [0, 30], 'color': '#D1FAE5'},
                            {'range': [30, 70], 'color': '#FEF3C7'},
                            {'range': [70, 100], 'color': '#FEE2E2'}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

# Page 5: Model Performance
elif selected_page == "📈 Model Performance":
    st.markdown("### Random Forest Classifier Performance Metrics")
    st.write(
        "Evaluate the technical accuracy, precision, and sensitivity (Recall) of the deployed Random Forest model. "
        "The model is validated against an independent, stratified test set representing 20% of the baseline cohort."
    )
    
    # Model KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #1A2B4C;">
            <div class="kpi-card-title">Test Accuracy</div>
            <div class="kpi-card-value">75.8%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #2A9D8F;">
            <div class="kpi-card-title">Retention Recall</div>
            <div class="kpi-card-value">78.3%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #E76F51;">
            <div class="kpi-card-title">Precision (Churn)</div>
            <div class="kpi-card-value">53.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #E9C46A;">
            <div class="kpi-card-title">ROC-AUC Score</div>
            <div class="kpi-card-value">84.2%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("#### Confusion Matrix Heatmap")
        
        # Create an interactive Plotly Heatmap for Confusion Matrix
        # [[775, 260], [81, 293]]
        cm = [[775, 260], [81, 293]]
        x_labels = ['Stay (0)', 'Churn (1)']
        y_labels = ['Stay (0)', 'Churn (1)']
        
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=x_labels, y=y_labels,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Class", y="Actual Class", color="Count")
        )
        fig_cm.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with m_col2:
        st.markdown("#### Operational Interpretation")
        st.write(
            "In customer retention, a **False Negative** (failing to identify a customer who is about to churn) is "
            "operationally and financially more costly than a **False Positive** (reaching out to a loyal customer with a retention discount). "
            "Because of this, we configure our Random Forest tree leaf costs to bias towards **Recall (78.3%)** rather than raw overall Accuracy."
        )
        st.write(
            "- **True Positives (293):** 293 churners were successfully identified. These are high-priority candidates for discount offers.\n"
            "- **True Negatives (775):** 775 loyal subscribers were correctly diagnosed as stable, avoiding unnecessary promotion expenses.\n"
            "- **False Positives (260):** 260 loyal customers were classified as at-risk. Giving them minor loyalty perks is a safe 'positive error'.\n"
            "- **False Negatives (81):** 81 customers were missed by the algorithm. These represent the leakage margin."
        )

# Page 6: Feature Importance
elif selected_page == "🧬 Feature Importance":
    st.markdown("### Top Factors Driving Customer Churn")
    st.write(
        "Feature importances describe the proportional influence of each variable on the Random Forest tree splits. "
        "To provide a meaningful business perspective, dummy columns created during categorical encoding "
        "are aggregated back into their original corporate groups."
    )
    
    model_path = "trained_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model.")
    else:
        pipeline = joblib.load(model_path)
        
        # Compute aggregated importances
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
        
        # Rename for visual styling
        rename_dict = {
            'Contract': 'Contract Type',
            'tenure': 'Tenure',
            'MonthlyCharges': 'Monthly Charges',
            'TotalCharges': 'Total Charges',
            'InternetService': 'Internet Service',
            'PaymentMethod': 'Payment Method',
            'Dependents': 'Dependents Status',
            'gender': 'Gender',
            'Partner': 'Partner Status',
            'SeniorCitizen': 'Senior Citizen Status',
            'PhoneService': 'Phone Service Status'
        }
        agg_imp_df['feature_name'] = agg_imp_df['feature'].map(rename_dict)
        
        # Plot Plotly bar chart
        fig_fi = px.bar(
            agg_imp_df.head(10).iloc[::-1], x='importance', y='feature_name',
            orientation='h', title="Top 10 Important Features (Aggregated)",
            color='importance', color_continuous_scale='Viridis'
        )
        fig_fi.update_layout(xaxis_title="Aggregated Gini Importance", yaxis_title="Feature Name", coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
        
        st.markdown("#### Strategic Insights from Variable Importance")
        st.write(
            "1. **Contract Type (26.0%):** The contractual commitment model is the single most dominant factor in subscriber churn. Transitioning customers off monthly terms is highly critical.\n"
            "2. **Tenure (16.6%):** Customer relationship age heavily affects loyalty. The first year is the most volatile period in the customer lifecycle.\n"
            "3. **Charges (Monthly: 14.0%, Total: 13.9%):** Billing volumes directly dictate customer lifetime value and attrition. Users with high monthly bills have strong price sensitivity.\n"
            "4. **Infrastructure (Internet Service: 12.0%):** The service type and speed tier affects churn, showing Fiber optic users are higher risk than DSL users."
        )

# Page 7: Business Insights
elif selected_page == "💡 Business Insights":
    st.markdown("### Executive Insights Panel")
    st.write(
        "The following statistical insights have been extracted from analyzing the subscriber base. "
        "These metrics represent core behavioral traits driving churn events."
    )
    
    insights = [
        ("Insight 1: Contract Term Vulnerability", 
         "Month-to-month contract customers represent <b>the highest risk profile</b>. Over 42% of month-to-month customers churned, "
         "whereas customers on 1-year or 2-year contract plans exhibited churn rates below 11% and 3% respectively. Month-to-month terms represent a low commitment trigger."),
        
        ("Insight 2: Early Relationship Attrition", 
         "Customers with low tenure (under 12 months) show massive churn probabilities. Attrition rates drop from "
         "<b>nearly 48% in month 1-6</b> down to under 8% for customers reaching a tenure of 24 months or longer. Retention efforts must focus heavily on early onboarding."),
        
        ("Insight 3: Monthly Billing Elasticity", 
         "Customers paying monthly fees exceeding $75 are <b>twice as likely to churn</b> than those paying lower billing rates. "
         "High fees provoke price comparison shopping, driving customer attrition to lower-cost competitors."),
        
        ("Insight 4: Digital Friction in Billing", 
         "Subscribers using <b>Electronic Check</b> payments exhibit a churn rate of 45%, compared to only 15-16% for "
         "customers using automated payment plans (Credit Card or Bank Auto-Pay). Manual monthly invoices create active checkpoints where users decide whether to continue services."),
        
        ("Insight 5: Product Deficiencies in High-Speed Tiers", 
         "Fiber optic customers exhibit a churn rate of <b>41.8%</b>, which is significantly higher than DSL users (18.9%). "
         "While Fiber offers superior tech speed, it also represents higher cost, suggesting service disruptions or billing expectations are prompting exits.")
    ]
    
    for title, desc in insights:
        st.markdown(f"""
        <div class="styled-list-item">
            <div class="styled-list-title">{title}</div>
            <div class="styled-list-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Page 8: Recommendations
elif selected_page == "🎯 Recommendations":
    st.markdown("### Strategic Retention Recommendations")
    st.write(
        "Operational strategies designed to counter churn indicators, optimize lifetime value, "
        "and maximize recurring revenues."
    )
    
    recommendations = [
        ("Recommendation 1: Contractual Migration Incentives", 
         "Implement a targeted contract migration program. Offer month-to-month subscribers a <b>10% billing discount</b> "
         "or a speed upgrade if they transition to an annual commitment. Locking in users to a contract completely offsets the minor margin loss by stabilizing long-term LTV."),
        
        ("Recommendation 2: Dedicated First-Year Onboarding Flow", 
         "Create a proactive customer success pipeline for the critical 0-12 month window. Establish automated service check-ins "
         "at day 30, 90, and 180. Providing priority tech support and onboarding guides during the first year reinforces product value and intercepts churn before it occurs."),
        
        ("Recommendation 3: Proactive Account Down-Sells", 
         "Monitor billing thresholds. When a user's monthly charges cross $75 and their usage remains moderate, proactively offer "
         "loyalty bundles or value-optimized tiers. It is far more profitable to retain a subscriber at a lower price tier than to lose them completely."),
        
        ("Recommendation 4: Incentivize Auto-Pay Enrollment", 
         "Launch an auto-pay campaign. Offer a one-time <b>$5 statement credit</b> to users who transition from Electronic Check "
         "to Automatic Credit Card or Bank draft payments. Moving customers to auto-pay completely removes manual invoice payment friction, which directly drives down voluntary churn."),
        
        ("Recommendation 5: Quality Assurance for Premium Tiers", 
         "Address Fiber Optic user churn by deploying specialized diagnostic checks. Audit service reliability and ensure that customers "
         "paying for high-speed fiber are receiving value. Targeted satisfaction surveys and billing transparent updates for high-speed users can address expectation gaps.")
    ]
    
    for title, desc in recommendations:
        st.markdown(f"""
        <div class="styled-list-item" style="border-left-color: #E76F51;">
            <div class="styled-list-title" style="color: #E76F51;">{title}</div>
            <div class="styled-list-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
