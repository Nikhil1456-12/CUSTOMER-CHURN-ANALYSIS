# Sales Forecasting & Demand Prediction Using Time Series Analysis

This repository contains a complete, production-ready Data Analytics and Business Intelligence (BI) web application built with **Streamlit** and **Facebook Prophet**. It analyzes historical sales data from the Kaggle Superstore Sales Dataset and projects future retail demand to support inventory planning and management decision making.

---

## 🌟 Application Features

The application consists of **9 structured pages** designed for retail managers and executives:

1. **🏠 Landing Page (Home)**: High-level introduction, dataset summary metrics, and business objectives.
2. **📊 Executive Dashboard**: High-level financial KPIs (Sales, Orders, Categories, Regions, Top Performers, and Out-of-Sample YoY growth forecasts) with interactive filters (Date range, Region, Category, Segment).
3. **📈 Sales Analytics**: Granular time-series views including Daily/Monthly trends, YoY overlays, moving averages, monthly average boxplots (seasonality), and transaction distributions.
4. **📦 Product Performance**: Best/Worst categories, top 10 products, revenue share timelines, and an interactive **Product Sales Treemap**.
5. **🗺️ Regional Analysis**: Sales by region, state rankings, and an interactive **US Choropleth Map** visualizing regional revenue density.
6. **🔮 Time Series Forecasting**: A Prophet model trained on daily sales with an interactive horizon selector (30, 60, and 90 days), out-of-sample forecasts (with 95% confidence bands), validation backtests, and component plots (growth trend, weekly, and yearly seasonalities).
7. **⚙️ Model Performance**: Detailed breakdown of accuracy metrics (MAE, RMSE, and MAPE) with a detailed statistical explanation of MAPE behavior on granular retail data.
8. **💡 Business Insights**: Structural metric blocks highlighting core data findings (Technology margins, West region growth, Q4 seasonality).
9. **📋 Management Recommendations**: Operational suggestions on inventory management, marketing allocations, warehouse stocking, and supply chain routing.
10. **📖 Project Documentation**: Formally structured project documentation detailing objectives, datasets, CRISP-DM methodology, modeling formulas, and future enhancements.

### 🚀 Advanced Features
* **Theme Selector**: Dynamic Dark/Light Mode toggle in the sidebar with matching CSS overrides and Plotly templates.
* **Download Forecast CSV**: Export forecasted demand data tables directly.
* **Download PDF Report**: Directly download a pre-compiled executive PDF report from the interface.
* **Streamlit Caching**: Uses `@st.cache_data` and `@st.cache_resource` to ensure instant page transitions and cached model training.

---

## 🛠️ Technology Stack
* **Frontend UI**: Streamlit, Custom HTML/CSS injection.
* **Backend processing**: Python, Pandas, NumPy, Scikit-learn.
* **Time Series Modeling**: Facebook Prophet.
* **Data Visualization**: Plotly Express & Graph Objects.

---

## 💻 Local Installation and Setup

Follow these steps to run the application locally on your machine:

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Sales-Forecasting-Demand-Prediction
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Cloud

To host this application publicly on Streamlit Cloud:

1. Push your local files to a public GitHub repository.
2. Visit [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**, select your repository, branch (`main`), and path to the main file (`app.py`).
4. Click **Deploy**. Streamlit Cloud will parse `requirements.txt`, install dependencies, and generate a public URL.

---

## 📁 Project Directory Structure
```
Sales-Forecasting-Demand-Prediction/
├── data/
│   └── train.csv           # Historical superstore sales data (9,800 rows)
├── report/
│   └── SALES_FORECASTING___DEMAND_PREDICTION_USING_TIME_SERIES_ANALYSIS.pdf # Pre-compiled report
├── app.py                  # Main Streamlit app source code
├── requirements.txt        # Python package requirements
├── README.md               # Project documentation guide
└── task.md                 # Project checklist (internal tracking)
```
