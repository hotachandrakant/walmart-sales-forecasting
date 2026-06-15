"""
01_data_cleaning.py
-------------------
Loads raw Walmart data, runs data-quality checks, produces EDA charts, and
writes a tidy data/walmart_clean.csv used downstream.

Usage:
    python 01_data_cleaning.py
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_walmart, money

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

df, source = load_walmart()

print("=" * 55)
print(f"  DATA CLEANING REPORT   (source: {source})")
print("=" * 55)
print(f"Rows        : {len(df):,}")
print(f"Stores      : {df['Store'].nunique()}")
print(f"Date range  : {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"Total sales : {money(df['Weekly_Sales'].sum())}")
print(f"Holiday weeks: {df['Holiday_Flag'].sum()} rows flagged")
print("\nMissing values per column:")
miss = df.isna().sum()
print(miss[miss > 0] if miss.sum() else "  none")

# ---------------- EDA charts ----------------
# 1. total weekly sales over time (all stores)
weekly = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
plt.figure(figsize=(12, 5))
plt.plot(weekly["Date"], weekly["Weekly_Sales"], color="#2563eb")
hol = df[df["Holiday_Flag"] == 1]["Date"].unique()
for h in hol:
    plt.axvline(h, color="#f59e0b", alpha=0.25)
plt.title("Total Weekly Sales Over Time (orange = holiday weeks)",
          fontsize=14, fontweight="bold")
plt.ylabel("Weekly Sales ($)")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
plt.tight_layout(); plt.savefig(f"{OUT}/01_sales_over_time.png"); plt.close()

# 2. holiday vs non-holiday average
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
hol_avg = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()
ax[0].bar(["Non-Holiday", "Holiday"], hol_avg.values, color=["#94a3b8", "#16a34a"])
ax[0].set_title("Avg Weekly Sales: Holiday vs Not", fontweight="bold")
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M"))
sns.histplot(df["Weekly_Sales"], kde=True, ax=ax[1], color="#2563eb")
ax[1].set_title("Distribution of Weekly Sales", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/02_holiday_distribution.png"); plt.close()

# 3. correlation of sales with economic factors
plt.figure(figsize=(7, 6))
cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]
sns.heatmap(df[cols].corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Correlation: Sales vs Economic Factors", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/03_correlation.png"); plt.close()

df.to_csv(f"{OUT}/walmart_clean.csv", index=False)
print("\nSaved EDA charts (01-03) + cleaned dataset -> outputs/walmart_clean.csv")
