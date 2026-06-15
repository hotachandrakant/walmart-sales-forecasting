"""
make_demo_data.py
-----------------
Creates a DEMO csv matching the EXACT schema of the Kaggle "Walmart Dataset"
(Walmart.csv), so this project runs out-of-the-box. Data has a trend, yearly
seasonality, holiday spikes, and economic regressors — just like the real file.

>>> Replace this with the REAL dataset for your submission:
    https://www.kaggle.com/datasets/yasserh/walmart-dataset
    Download "Walmart.csv" and put it in this same data/ folder.
    The loader automatically prefers the real file if it is present.

Usage:
    python make_demo_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_STORES = 12
# 143 weekly dates, same span as the real Walmart data
dates = pd.date_range("2010-02-05", periods=143, freq="W-FRI")

# Holiday weeks (Super Bowl, Labour Day, Thanksgiving, Christmas) -> flag = 1
holiday_weeks = set()
for d in dates:
    if (d.month == 2 and 7 <= d.day <= 13):          # Super Bowl
        holiday_weeks.add(d)
    if (d.month == 9 and 6 <= d.day <= 12):          # Labour Day
        holiday_weeks.add(d)
    if (d.month == 11 and 22 <= d.day <= 28):        # Thanksgiving
        holiday_weeks.add(d)
    if (d.month == 12 and 24 <= d.day <= 31):        # Christmas
        holiday_weeks.add(d)

rows = []
for store in range(1, N_STORES + 1):
    base = RNG.uniform(0.8e6, 1.8e6)                 # store size baseline
    trend_slope = RNG.uniform(-200, 600)             # per-week trend
    for i, d in enumerate(dates):
        doy = d.dayofyear
        yearly = 1 + 0.18 * np.sin(2 * np.pi * (doy - 60) / 365)   # seasonal
        # Strong Nov/Dec lift
        if d.month in (11, 12):
            yearly *= 1.25
        holiday = d in holiday_weeks
        holiday_boost = 1.18 if holiday else 1.0
        noise = RNG.normal(1, 0.05)
        sales = round((base + trend_slope * i) * yearly * holiday_boost * noise, 2)

        temperature = round(60 + 25 * np.sin(2 * np.pi * (doy - 110) / 365)
                            + RNG.normal(0, 4), 2)     # deg F
        fuel = round(2.6 + 0.5 * (i / len(dates)) + RNG.normal(0, 0.05), 2)
        cpi = round(210 + 8 * (i / len(dates)) + RNG.normal(0, 0.3), 3)
        unemp = round(8.2 - 1.2 * (i / len(dates)) + RNG.normal(0, 0.1), 3)

        rows.append({
            "Store": store,
            "Date": d.strftime("%d-%m-%Y"),            # real file uses dd-mm-yyyy
            "Weekly_Sales": sales,
            "Holiday_Flag": int(holiday),
            "Temperature": temperature,
            "Fuel_Price": fuel,
            "CPI": cpi,
            "Unemployment": unemp,
        })

df = pd.DataFrame(rows)
df.to_csv("data/sample_demo_data.csv", index=False)
print(f"Demo data written: data/sample_demo_data.csv ({len(df):,} rows, "
      f"{N_STORES} stores x {len(dates)} weeks)")
print(f"Total sales: ${df.Weekly_Sales.sum():,.0f}")
print("NOTE: replace with the real Kaggle 'Walmart.csv' for submission.")
