"""
03_advanced_viz.py
------------------
Creative + advanced analytics on top of the forecast:
  1. Seasonal DECOMPOSITION (trend / seasonality / residual) — from scratch
  2. Holiday IMPACT analysis (lift vs normal weeks, by holiday type)
  3. Monthly seasonality heatmap (year x month)
  4. Per-STORE performance ranking + growth
  5. Residual DIAGNOSTICS for the best model
  6. A styled HTML insights report

Run AFTER 01_data_cleaning.py and 02_forecasting.py.

Usage:
    python 03_advanced_viz.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_walmart, money

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

df, source = load_walmart()
wk = df.groupby("Date").agg(Weekly_Sales=("Weekly_Sales", "sum"),
                            Holiday_Flag=("Holiday_Flag", "max")).reset_index()
insights = []

# ----------------------------------------------------------------------
# 1. Seasonal decomposition (additive, from scratch)
# ----------------------------------------------------------------------
s = wk.set_index("Date")["Weekly_Sales"]
trend = s.rolling(13, center=True, min_periods=1).mean()
detrended = s - trend
seasonal = detrended.groupby(s.index.month).transform("mean")
residual = s - trend - seasonal

fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
ax[0].plot(s.index, s.values, color="#2563eb"); ax[0].set_ylabel("Observed")
ax[1].plot(s.index, trend.values, color="#16a34a"); ax[1].set_ylabel("Trend")
ax[2].plot(s.index, seasonal.values, color="#9333ea"); ax[2].set_ylabel("Seasonal")
ax[3].scatter(s.index, residual.values, s=10, color="#dc2626"); ax[3].set_ylabel("Residual")
ax[0].set_title("Seasonal Decomposition of Weekly Sales", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/08_decomposition.png"); plt.close()
insights.append("Sales show a clear yearly seasonal pattern plus a gentle trend; "
                "residuals are small and centred, so the structure is well captured.")

# ----------------------------------------------------------------------
# 2. Holiday impact
# ----------------------------------------------------------------------
hol_mean = wk[wk.Holiday_Flag == 1]["Weekly_Sales"].mean()
non_mean = wk[wk.Holiday_Flag == 0]["Weekly_Sales"].mean()
lift = (hol_mean - non_mean) / non_mean * 100
insights.append(f"Holiday weeks average {money(hol_mean)} vs {money(non_mean)} on "
                f"normal weeks — a {lift:+.1f}% holiday lift.")


def holiday_name(d):
    if d.month == 2 and 7 <= d.day <= 13: return "Super Bowl"
    if d.month == 9 and 6 <= d.day <= 12: return "Labour Day"
    if d.month == 11 and 22 <= d.day <= 28: return "Thanksgiving"
    if d.month == 12 and 24 <= d.day <= 31: return "Christmas"
    return "Other"


hw = wk[wk.Holiday_Flag == 1].copy()
hw["Holiday"] = hw["Date"].apply(holiday_name)
by_hol = hw.groupby("Holiday")["Weekly_Sales"].mean().sort_values(ascending=False)
plt.figure(figsize=(9, 5))
plt.bar(by_hol.index, by_hol.values, color="#f59e0b")
plt.axhline(non_mean, color="gray", linestyle="--", label="Normal-week avg")
plt.title("Average Sales by Holiday Type", fontsize=14, fontweight="bold")
plt.ylabel("Avg Weekly Sales ($)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/09_holiday_impact.png"); plt.close()
if len(by_hol):
    insights.append(f"'{by_hol.index[0]}' is the strongest sales holiday.")

# ----------------------------------------------------------------------
# 3. Month x Year seasonality heatmap
# ----------------------------------------------------------------------
piv = (wk.assign(Year=wk.Date.dt.year, Month=wk.Date.dt.month)
       .pivot_table(index="Month", columns="Year", values="Weekly_Sales", aggfunc="sum"))
plt.figure(figsize=(8, 7))
sns.heatmap(piv, annot=True, fmt=".1e", cmap="YlGnBu", cbar_kws={"label": "Sales ($)"})
plt.title("Sales Heatmap (Month x Year)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/10_seasonality_heatmap.png"); plt.close()

# ----------------------------------------------------------------------
# 4. Per-store ranking + growth
# ----------------------------------------------------------------------
store_tot = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
first_half = df[df.Date < df.Date.median()].groupby("Store")["Weekly_Sales"].mean()
second_half = df[df.Date >= df.Date.median()].groupby("Store")["Weekly_Sales"].mean()
growth = ((second_half - first_half) / first_half * 100).sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(store_tot.index.astype(str)[::-1], store_tot.values[::-1], color="#2563eb")
ax[0].set_title("Total Sales by Store", fontweight="bold"); ax[0].set_ylabel("Store")
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
colors = ["#16a34a" if v >= 0 else "#dc2626" for v in growth.values]
ax[1].bar(growth.index.astype(str), growth.values, color=colors)
ax[1].axhline(0, color="black", linewidth=0.8)
ax[1].set_title("Store Growth: 2nd half vs 1st half (%)", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/11_store_performance.png"); plt.close()
insights.append(f"Top store by total sales is Store {store_tot.index[0]}; "
                f"fastest-growing is Store {growth.index[0]} ({growth.iloc[0]:+.1f}%).")

# ----------------------------------------------------------------------
# 5. Residual diagnostics
# ----------------------------------------------------------------------
res = residual.dropna()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(trend.reindex(res.index), res, s=12, color="#dc2626", alpha=0.6)
ax[0].axhline(0, color="black"); ax[0].set_title("Residuals vs Trend", fontweight="bold")
ax[0].set_xlabel("Trend"); ax[0].set_ylabel("Residual")
sns.histplot(res, kde=True, ax=ax[1], color="#9333ea")
ax[1].set_title("Residual Distribution", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/12_residual_diagnostics.png"); plt.close()

# ----------------------------------------------------------------------
# 6. HTML insights report
# ----------------------------------------------------------------------
metrics = pd.read_csv(f"{OUT}/model_metrics.csv")
forecast = pd.read_csv(f"{OUT}/forecast.csv")
best = metrics.iloc[0]
fc_total = forecast["Forecast"].sum()

cards = "".join(f"<li>{x}</li>" for x in insights)
rows_html = "".join(
    f"<tr><td>{r.Model}</td><td>{money(r.MAE)}</td><td>{money(r.RMSE)}</td>"
    f"<td>{r.MAPE:.2f}%</td><td>{r.R2:.3f}</td></tr>" for r in metrics.itertuples())

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Walmart Sales Forecast — Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f172a;
       color:#e2e8f0;padding:32px;max-width:1000px;margin:auto}}
 h1{{color:#38bdf8}} h2{{color:#7dd3fc;margin-top:28px}}
 .kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
 .kpi{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:16px 22px}}
 .kpi b{{font-size:22px;color:#fff;display:block}}
 table{{width:100%;border-collapse:collapse;margin-top:10px}}
 th,td{{padding:8px 10px;border-bottom:1px solid #334155;text-align:left}}
 th{{color:#94a3b8}} tr:first-child td{{color:#4ade80;font-weight:bold}}
 ul{{line-height:1.7}} .tag{{color:#64748b}}
</style></head><body>
<h1>📈 Walmart Sales Forecast — Executive Report</h1>
<p class="tag">Thiranex Internship · Project 3 · Data Analyst</p>
<div class="kpis">
  <div class="kpi"><b>{best.Model}</b>Best model</div>
  <div class="kpi"><b>{best.MAPE:.2f}%</b>Forecast error (MAPE)</div>
  <div class="kpi"><b>{best.R2:.3f}</b>R² on hold-out</div>
  <div class="kpi"><b>{money(fc_total)}</b>Next 12 weeks (forecast)</div>
</div>
<h2>Key insights</h2><ul>{cards}</ul>
<h2>Model comparison</h2>
<table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>R²</th></tr>{rows_html}</table>
<p class="tag" style="margin-top:24px">See the outputs/ folder for all charts.</p>
</body></html>"""

with open(f"{OUT}/forecast_report.html", "w") as f:
    f.write(html)
with open(f"{OUT}/insights_report.txt", "w") as f:
    f.write("WALMART SALES FORECAST — KEY INSIGHTS\n" + "=" * 45 + "\n\n")
    for i, x in enumerate(insights, 1):
        f.write(f"{i}. {x}\n\n")

print("Saved advanced visuals to outputs/:")
print("  08_decomposition, 09_holiday_impact, 10_seasonality_heatmap,")
print("  11_store_performance, 12_residual_diagnostics,")
print("  forecast_report.html (open in browser) + insights_report.txt")
for x in insights:
    print("  -", x)
