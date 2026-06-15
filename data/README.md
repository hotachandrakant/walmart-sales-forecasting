# 📥 Dataset — Kaggle "Walmart Dataset"

This project uses the **Walmart Dataset** from Kaggle for sales forecasting.

## 🔗 Dataset link
**https://www.kaggle.com/datasets/yasserh/walmart-dataset**

## ⬇️ How to download and use it
1. Open the link above (sign in to Kaggle — free).
2. Click **Download**. You get a zip.
3. Unzip it — you'll find **`Walmart.csv`**.
4. Put that file **into this `data/` folder** keeping the name `Walmart.csv`.
5. Re-run the scripts. The loader (`utils.py`) automatically prefers the real
   Kaggle file over the demo file (and handles its dd-mm-yyyy date format).

```bash
cd project-3-predictive-analytics
python 01_data_cleaning.py
python 02_forecasting.py
python 03_advanced_viz.py
streamlit run dashboard.py
```

## 📋 Dataset schema (8 columns, ~6,435 rows = 45 stores x 143 weeks)
| Column | Description |
|--------|-------------|
| Store | Store number (1–45) |
| Date | Week (dd-mm-yyyy) |
| **Weekly_Sales** | Sales for that store that week (USD) — the target |
| Holiday_Flag | 1 if the week has a major holiday, else 0 |
| Temperature | Average regional temperature (°F) |
| Fuel_Price | Regional fuel cost |
| CPI | Consumer Price Index |
| Unemployment | Regional unemployment rate |

The four holidays flagged are **Super Bowl, Labour Day, Thanksgiving, Christmas**
— the weeks that drive the biggest sales spikes.

> A demo file (`sample_demo_data.csv`) with the **same schema** is included so
> the project runs before you download the real one. Replace it for your
> actual submission.
