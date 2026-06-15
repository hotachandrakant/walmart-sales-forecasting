"""
02_forecasting.py
-----------------
Advanced sales forecasting with MULTIPLE models, honest time-based evaluation,
recursive multi-step forecasting, and confidence intervals.

Pipeline:
  1. Aggregate to company-wide weekly sales + weekly economic factors
  2. Feature engineering: calendar parts, lags (1/2/52), rolling means, holiday,
     and economic regressors (Temperature, Fuel, CPI, Unemployment)
  3. Time-based train/test split (no shuffling)
  4. Compare 4 models:
       Linear Regression, Random Forest, Gradient Boosting,
       and a from-scratch Holt-Winters (triple exponential smoothing)
  5. Pick the best by MAPE, recursively forecast the next 12 weeks
     with a 95% confidence band (from test residuals)
  6. Save charts, metrics, the forecast CSV, and the trained model

Usage:
    python 02_forecasting.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_walmart, holt_winters_additive, money

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def mape(y, p):
    y, p = np.array(y), np.array(p)
    return np.mean(np.abs((y - p) / y)) * 100


df, source = load_walmart()
print(f"Loaded {len(df):,} rows (source: {source})")

# ----------------------------------------------------------------------
# 1. Aggregate to company-wide weekly series
# ----------------------------------------------------------------------
wk = df.groupby("Date").agg(
    Weekly_Sales=("Weekly_Sales", "sum"),
    Temperature=("Temperature", "mean"),
    Fuel_Price=("Fuel_Price", "mean"),
    CPI=("CPI", "mean"),
    Unemployment=("Unemployment", "mean"),
    Holiday_Flag=("Holiday_Flag", "max"),
).reset_index().sort_values("Date").reset_index(drop=True)

# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------
wk["Month"] = wk["Date"].dt.month
wk["Week"] = wk["Date"].dt.isocalendar().week.astype(int)
wk["t"] = np.arange(len(wk))
for lag in (1, 2, 52):
    wk[f"lag{lag}"] = wk["Weekly_Sales"].shift(lag)
wk["roll4"] = wk["Weekly_Sales"].shift(1).rolling(4).mean()
wk["roll12"] = wk["Weekly_Sales"].shift(1).rolling(12).mean()
model_df = wk.dropna().reset_index(drop=True)

FEATURES = ["t", "Month", "Week", "Holiday_Flag", "Temperature", "Fuel_Price",
            "CPI", "Unemployment", "lag1", "lag2", "lag52", "roll4", "roll12"]
X, y = model_df[FEATURES], model_df["Weekly_Sales"]

# ----------------------------------------------------------------------
# 3. Time-based split (last 20% as test)
# ----------------------------------------------------------------------
split = int(len(model_df) * 0.8)
Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
print(f"Train weeks: {len(Xtr)}  |  Test weeks: {len(Xte)}")

# ----------------------------------------------------------------------
# 4. Compare models
# ----------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
}
results, rows = {}, []
for name, m in models.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    results[name] = {"model": m, "pred": pred}
    rows.append({"Model": name, "MAE": mean_absolute_error(yte, pred),
                 "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                 "MAPE": mape(yte, pred), "R2": r2_score(yte, pred)})

# Holt-Winters (univariate) on the raw aggregated series
hw_train = wk["Weekly_Sales"].iloc[:split + (len(wk) - len(model_df))]
hw_fc = holt_winters_additive(hw_train, season_len=52, n_forecast=len(yte))
rows.append({"Model": "Holt-Winters", "MAE": mean_absolute_error(yte, hw_fc),
             "RMSE": np.sqrt(mean_squared_error(yte, hw_fc)),
             "MAPE": mape(yte, hw_fc), "R2": r2_score(yte, hw_fc)})
results["Holt-Winters"] = {"model": None, "pred": np.array(hw_fc)}

metrics = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
metrics.round(2).to_csv(f"{OUT}/model_metrics.csv", index=False)
print("\n--- Model Comparison (time-based hold-out) ---")
print(metrics.assign(MAE=lambda d: d.MAE.map(money), RMSE=lambda d: d.RMSE.map(money),
                     MAPE=lambda d: d.MAPE.round(2).astype(str) + "%",
                     R2=lambda d: d.R2.round(3)).to_string(index=False))

best_name = metrics.iloc[0]["Model"]
print(f"\nBest model: {best_name}")

# bar chart of metrics
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].bar(metrics["Model"], metrics["MAPE"], color="#2563eb")
ax[0].set_title("MAPE by Model (lower = better)", fontweight="bold")
ax[0].set_ylabel("MAPE %"); ax[0].tick_params(axis="x", rotation=20)
ax[1].bar(metrics["Model"], metrics["R2"], color="#16a34a")
ax[1].set_title("R² by Model (higher = better)", fontweight="bold")
ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{OUT}/04_model_comparison.png"); plt.close()

# actual vs predicted on test window
plt.figure(figsize=(12, 5))
test_dates = model_df["Date"].iloc[split:]
plt.plot(model_df["Date"], model_df["Weekly_Sales"], color="#1e293b", label="Actual")
for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
    plt.plot(test_dates, results[name]["pred"], "--", label=f"{name}")
plt.title("Actual vs Predicted — test window", fontsize=14, fontweight="bold")
plt.ylabel("Weekly Sales ($)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f"{OUT}/05_actual_vs_predicted.png"); plt.close()

# ----------------------------------------------------------------------
# 5. Recursive 12-week forecast with the best ML model (+ CI)
# ----------------------------------------------------------------------
N_FORECAST = 12
if best_name == "Holt-Winters":
    fc_vals = holt_winters_additive(wk["Weekly_Sales"], season_len=52,
                                    n_forecast=N_FORECAST)
    best_model = None
else:
    best_model = results[best_name]["model"]
    best_model.fit(X, y)                              # refit on all data
    hist = wk.copy()
    series = hist["Weekly_Sales"].tolist()
    last_date = hist["Date"].iloc[-1]
    fc_vals = []
    for step in range(1, N_FORECAST + 1):
        d = last_date + pd.Timedelta(weeks=step)
        feat = {
            "t": len(hist) + step - 1,
            "Month": d.month,
            "Week": int(d.isocalendar().week),
            "Holiday_Flag": int((d.month == 11 and 22 <= d.day <= 28) or
                                (d.month == 12 and 24 <= d.day <= 31) or
                                (d.month == 2 and 7 <= d.day <= 13) or
                                (d.month == 9 and 6 <= d.day <= 12)),
            "Temperature": hist["Temperature"].iloc[-1],
            "Fuel_Price": hist["Fuel_Price"].iloc[-1],
            "CPI": hist["CPI"].iloc[-1],
            "Unemployment": hist["Unemployment"].iloc[-1],
            "lag1": series[-1],
            "lag2": series[-2],
            "lag52": series[-52] if len(series) >= 52 else series[-1],
            "roll4": np.mean(series[-4:]),
            "roll12": np.mean(series[-12:]),
        }
        yhat = float(best_model.predict(pd.DataFrame([feat])[FEATURES])[0])
        fc_vals.append(yhat)
        series.append(yhat)

# 95% CI from test residuals of the best model
resid_std = np.std(np.array(yte) - np.array(results[best_name]["pred"]))
ci = 1.96 * resid_std
future_dates = [wk["Date"].iloc[-1] + pd.Timedelta(weeks=s) for s in range(1, N_FORECAST + 1)]
forecast = pd.DataFrame({
    "Date": future_dates,
    "Forecast": np.round(fc_vals, 2),
    "Lower_95": np.round(np.array(fc_vals) - ci, 2),
    "Upper_95": np.round(np.array(fc_vals) + ci, 2),
})
forecast.to_csv(f"{OUT}/forecast.csv", index=False)
print(f"\n--- {N_FORECAST}-Week Forecast ({best_name}) ---")
for _, r in forecast.iterrows():
    print(f"{r['Date'].date()} : {money(r['Forecast'])}  "
          f"[{money(r['Lower_95'])} – {money(r['Upper_95'])}]")

# forecast chart with CI band
plt.figure(figsize=(12, 5))
plt.plot(wk["Date"], wk["Weekly_Sales"], color="#2563eb", label="Historical")
plt.plot(forecast["Date"], forecast["Forecast"], color="#dc2626", marker="o",
         linestyle="--", label="Forecast")
plt.fill_between(forecast["Date"], forecast["Lower_95"], forecast["Upper_95"],
                 color="#dc2626", alpha=0.15, label="95% CI")
plt.axvline(wk["Date"].iloc[-1], color="gray", linestyle=":")
plt.title(f"Company Weekly Sales Forecast — next {N_FORECAST} weeks ({best_name})",
          fontsize=14, fontweight="bold")
plt.ylabel("Weekly Sales ($)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f"{OUT}/06_forecast.png"); plt.close()

# feature importance (Random Forest, always available)
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(9, 6))
plt.barh(imp.index, imp.values, color="#16a34a")
plt.title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig(f"{OUT}/07_feature_importance.png"); plt.close()

# save model bundle
joblib.dump({"best_name": best_name, "model": best_model, "features": FEATURES,
             "ci": ci, "history_tail": wk.tail(60)},
            f"{OUT}/forecast_model.joblib")

print(f"\nSaved charts + model_metrics.csv + forecast.csv + "
      f"forecast_model.joblib to '{OUT}/'")
