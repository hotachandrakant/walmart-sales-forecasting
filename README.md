# 📈 Project 3 — Sales Forecasting with Predictive Analytics (Advanced + Creative)

**Thiranex Internship — Data Analyst Track**

Forecast future Walmart weekly sales from historical data: clean & engineer
features, compare four forecasting models with honest time-based validation,
project the next weeks with confidence intervals, and explore everything in an
interactive 4-tab dashboard.

> 🔗 **Dataset:** https://www.kaggle.com/datasets/yasserh/walmart-dataset
> (download `Walmart.csv` into `data/` — see `data/README.md`)

---

## ⭐ What makes this advanced + creative
**Advanced modelling**
- **4 models compared** — Linear Regression, Random Forest, Gradient Boosting,
  and a **from-scratch Holt-Winters** (triple exponential smoothing) coded by hand
- **Time-series feature engineering** — calendar parts, lags (1 / 2 / 52 weeks),
  rolling means, holiday flag, and economic regressors (Temperature, Fuel, CPI, Unemployment)
- **Honest time-based split** (no shuffling) + 4 metrics: **MAE, RMSE, MAPE, R²**
- **Recursive multi-step forecast** with a **95% confidence band** from residuals
- **Feature importance** + a saved model bundle (`forecast_model.joblib`)

**Creative analytics**
- **Seasonal decomposition** (trend / seasonality / residual) built from scratch
- **Holiday impact** analysis — lift vs normal weeks, ranked by holiday type
- **Month × Year seasonality heatmap**
- **Per-store ranking + growth** comparison
- **Residual diagnostics** (residual vs trend + distribution)
- **Styled HTML executive report** (`forecast_report.html`)

**Interactive dashboard (4 tabs)**
1. 📈 **Trends** — company or per-store sales with holiday markers
2. 🔮 **Forecast** — pick model + horizon, see forecast with a 95% CI band
3. 🎉 **Holidays** — holiday lift + seasonality heatmap
4. 🏬 **Stores** — store ranking + single-store deep dive

## 📊 Results (demo run)
| Model | MAPE | R² |
|-------|------|----|
| **Linear Regression** ✅ | ~1.5% | ~0.97 |
| Holt-Winters (from scratch) | ~1.9% | ~0.96 |
| Random Forest | ~1.9% | ~0.95 |
| Gradient Boosting | ~2.0% | ~0.95 |

The forecast correctly spikes around **Thanksgiving and Christmas** weeks.

## 🗂 File structure
```
project-3-predictive-analytics/
├── README.md
├── requirements.txt
├── utils.py                  # loader + feature engineering + Holt-Winters
├── make_demo_data.py         # creates a schema-matching demo CSV
├── 01_data_cleaning.py       # quality report + EDA charts -> walmart_clean.csv
├── 02_forecasting.py         # multi-model compare + forecast + CI + model save
├── 03_advanced_viz.py        # decomposition, holiday, heatmap, stores, report
├── dashboard.py              # 4-tab interactive Streamlit app
├── data/
│   ├── README.md             # Kaggle link + schema
│   └── sample_demo_data.csv  # demo data (replace with real Walmart.csv)
└── outputs/
    ├── 01_sales_over_time.png ... 12_residual_diagnostics.png  (charts)
    ├── model_metrics.csv
    ├── forecast.csv
    ├── walmart_clean.csv
    ├── forecast_model.joblib
    ├── forecast_report.html      # open in a browser
    └── insights_report.txt
```

## ▶️ How to run
```bash
# 1. install
pip install -r requirements.txt

# 2. add data (real Kaggle file recommended) OR generate demo:
python make_demo_data.py

# 3. run the pipeline
python 01_data_cleaning.py     # clean + EDA
python 02_forecasting.py       # compare models, forecast, save model
python 03_advanced_viz.py      # decomposition / holiday / stores / report

# 4. launch the dashboard
streamlit run dashboard.py
```

## 🛠 Tech stack
Python · pandas · NumPy · scikit-learn (LinearRegression, RandomForest,
GradientBoosting, metrics) · custom Holt-Winters · SciPy · joblib ·
Matplotlib · Seaborn · Plotly · Streamlit

## 📚 What I learned
Time-series feature engineering, comparing forecasting models with proper
time-based validation, multi-step recursive forecasting with confidence
intervals, seasonal decomposition, and turning forecasts into a business report.
