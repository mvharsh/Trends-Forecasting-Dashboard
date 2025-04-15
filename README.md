# 📊 Reliance Trends Madurai - Sales Analytics & Forecasting Dashboard

A data-driven, interactive Streamlit dashboard that analyzes and forecasts monthly sales revenue for Reliance Trends' Madurai store. This project empowers business stakeholders with detailed insights into purchase behaviors, product trends, payment preferences, and future revenue predictions using machine learning.

---

## 📁 Dataset Overview

The dataset used is `Reliance_Trends_Madurai_Preprocessed.csv`, which includes:

- **Date of purchase**
- **Purchase amount in USD**
- **Product category**
- **Customer gender**
- **Store location**
- **Payment method**
- **Review ratings**
- **Product names**

It has been preprocessed to include additional time-based features such as:

- `day_of_week`
- `month_name`
- `year`, `month`
- Rolling and lag features for time series modeling

---

## 🧹 Data Preprocessing

Key preprocessing steps:

1. **Datetime conversion** for accurate time-based grouping.
2. **Aggregation**: Monthly sales grouped to derive revenue trends.
3. **Feature engineering**:
   - Revenue lags: `revenue_lag_1`, `revenue_lag_2`, `revenue_lag_3`
   - Rolling mean: `revenue_rolling_mean_3`
4. **Filtering UI**: Streamlit sidebar filters for:
   - Date range
   - Product categories
   - Store locations
   - Payment methods

---

## 🤖 Model Building & Evaluation

### Models Tried:
- **XGBoost Regressor**
- **Gradient Boosting Regressor (Selected as Final Model)**

### Final Model:
```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
```

### Train-Test Split:
- 80% training, 20% testing
- No shuffling to preserve time series order

### Evaluation Metrics:
- **R² Score**
- **RMSE**
- **Forecast error percentage**

---

## 🔮 Forecasting Approach

- Recursive multi-step forecasting using the last 3 months of data and rolling mean
- Forecast horizon: Configurable (1 to 12 months)
- Future predictions are visualized along with historical actual and test predictions

---

## 📊 Dashboard Features

### Tabs:
- **📈 Sales Analysis**
  - Monthly revenue trends
  - Category-wise and payment method-wise revenue
  - Day-of-week patterns
- **🔮 Forecasting**
  - Model performance
  - Forecasted future revenue
  - Month-on-month prediction details
- **📊 Advanced Metrics**
  - Monthly growth rate
  - Heatmap of sales by month and year
  - Sales by location
  - Gender-based purchase distribution
  - Top purchased items
  - Customer review rating analysis

### Sidebar Filters:
- Time range
- Product categories
- Store locations
- Payment methods
- Forecast months

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/reliance-trends-sales-dashboard.git
cd reliance-trends-sales-dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset (`Reliance_Trends_Madurai_Preprocessed.csv`) in the project folder.

4. Run the app:

```bash
streamlit run app.py
```

---

## 📦 Dependencies

See `requirements.txt` for a full list. Key packages include:

- `pandas`, `numpy`
- `scikit-learn`, `xgboost`
- `streamlit`
- `plotly`, `matplotlib`

---
## 📊 Dashboard (Deployed in Streamlit) 

https://reliance-trends-forecasting-dashboard.streamlit.app/

---

## 🔗 Blog Link 

https://medium.com/@sweathasm.mohan07/sales-overview-dashboard-f94e07323297
