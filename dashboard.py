"""
dashboard.py
------------
Interactive sales-forecasting app (Streamlit + Plotly).

4 tabs:
  📈 Trends      - company / per-store sales over time, holiday markers
  🔮 Forecast    - pick horizon + model, see forecast with a 95% confidence band
  🎉 Holidays    - holiday lift analysis + seasonality heatmap
  🏬 Stores      - store ranking, growth, and a single-store deep dive

Run:
    pip install -r requirements.txt
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import load_walmart, holt_winters_additive, money

st.set_page_config(page_title="Sales Forecasting", layout="wide", page_icon="📈")


@st.cache_data
def get_data():
    return load_walmart()


df, source = get_data()

st.sidebar.header("⚙️ Settings")
store_opt = ["All stores (company)"] + [f"Store {s}" for s in sorted(df["Store"].unique())]
sel_store = st.sidebar.selectbox("Scope", store_opt)
model_name = st.sidebar.selectbox("Forecast model",
                                  ["Linear Regression", "Random Forest",
                                   "Gradient Boosting", "Holt-Winters"])
horizon = st.sidebar.slider("Forecast horizon (weeks)", 4, 26, 12)
st.sidebar.caption(f"Data source: {source}")

# Build the working weekly series for the chosen scope
if sel_store == "All stores (company)":
    base = df.groupby("Date").agg(Weekly_Sales=("Weekly_Sales", "sum"),
                                  Holiday_Flag=("Holiday_Flag", "max")).reset_index()
else:
    sid = int(sel_store.split()[1])
    base = (df[df["Store"] == sid].groupby("Date")
            .agg(Weekly_Sales=("Weekly_Sales", "sum"),
                 Holiday_Flag=("Holiday_Flag", "max")).reset_index())
base = base.sort_values("Date").reset_index(drop=True)

st.title("📈 Sales Forecasting Dashboard")
st.caption("Thiranex Internship — Project 3 (Advanced) | Data Analyst")

c = st.columns(4)
c[0].metric("Scope", sel_store.replace("All stores (company)", "Company"))
c[1].metric("Weeks of history", len(base))
c[2].metric("Total sales", money(base["Weekly_Sales"].sum()))
c[3].metric("Avg weekly", money(base["Weekly_Sales"].mean()))

t1, t2, t3, t4 = st.tabs(["📈 Trends", "🔮 Forecast", "🎉 Holidays", "🏬 Stores"])

# ===== Trends =====
with t1:
    fig = px.line(base, x="Date", y="Weekly_Sales", title="Weekly Sales Over Time")
    for h in base[base["Holiday_Flag"] == 1]["Date"]:
        fig.add_vline(x=h, line_dash="dot", line_color="orange", opacity=0.3)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange dotted lines mark holiday weeks.")

# ===== Forecast =====
with t2:
    series = base.copy()
    series["t"] = np.arange(len(series))
    series["Month"] = series["Date"].dt.month
    for lag in (1, 2, 52):
        series[f"lag{lag}"] = series["Weekly_Sales"].shift(lag)
    series["roll4"] = series["Weekly_Sales"].shift(1).rolling(4).mean()
    m = series.dropna().reset_index(drop=True)
    feats = ["t", "Month", "Holiday_Flag", "lag1", "lag2", "lag52", "roll4"]

    if model_name == "Holt-Winters":
        fc = holt_winters_additive(base["Weekly_Sales"], season_len=52,
                                   n_forecast=horizon)
        resid_std = base["Weekly_Sales"].std() * 0.1
    else:
        mdl = {"Linear Regression": LinearRegression(),
               "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
               "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
               }[model_name]
        mdl.fit(m[feats], m["Weekly_Sales"])
        vals = base["Weekly_Sales"].tolist()
        fc, last_date = [], base["Date"].iloc[-1]
        for step in range(1, horizon + 1):
            d = last_date + pd.Timedelta(weeks=step)
            row = {"t": len(base) + step - 1, "Month": d.month,
                   "Holiday_Flag": int(d.month in (11, 12) and d.day >= 22),
                   "lag1": vals[-1], "lag2": vals[-2],
                   "lag52": vals[-52] if len(vals) >= 52 else vals[-1],
                   "roll4": np.mean(vals[-4:])}
            yhat = float(mdl.predict(pd.DataFrame([row])[feats])[0])
            fc.append(yhat); vals.append(yhat)
        resid_std = np.std(m["Weekly_Sales"] - mdl.predict(m[feats]))

    ci = 1.96 * resid_std
    fdates = [base["Date"].iloc[-1] + pd.Timedelta(weeks=s) for s in range(1, horizon + 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base["Date"], y=base["Weekly_Sales"],
                             name="Historical", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=fdates, y=fc, name="Forecast",
                             line=dict(color="#dc2626", dash="dash")))
    fig.add_trace(go.Scatter(x=fdates + fdates[::-1],
                             y=list(np.array(fc) + ci) + list((np.array(fc) - ci)[::-1]),
                             fill="toself", fillcolor="rgba(220,38,38,0.15)",
                             line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig.update_layout(title=f"{horizon}-week forecast — {model_name}")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Forecasted total (next %d weeks)" % horizon, money(sum(fc)))

# ===== Holidays =====
with t3:
    col1, col2 = st.columns(2)
    with col1:
        hol = base.groupby("Holiday_Flag")["Weekly_Sales"].mean()
        comp = pd.DataFrame({"Type": ["Non-Holiday", "Holiday"],
                             "Avg": [hol.get(0, 0), hol.get(1, 0)]})
        st.plotly_chart(px.bar(comp, x="Type", y="Avg", color="Type",
                               title="Avg Sales: Holiday vs Normal",
                               color_discrete_sequence=["#94a3b8", "#16a34a"]),
                        use_container_width=True)
    with col2:
        hm = base.assign(Year=base.Date.dt.year, Month=base.Date.dt.month)
        piv = hm.pivot_table(index="Month", columns="Year",
                             values="Weekly_Sales", aggfunc="sum")
        st.plotly_chart(px.imshow(piv, title="Sales Heatmap (Month x Year)",
                                  color_continuous_scale="YlGnBu", aspect="auto"),
                        use_container_width=True)

# ===== Stores =====
with t4:
    tot = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(tot, x="Store", y="Weekly_Sales", title="Total Sales by Store",
                           color="Weekly_Sales", color_continuous_scale="Blues"),
                    use_container_width=True)
    pick = st.selectbox("Deep-dive into a store", sorted(df["Store"].unique()))
    sd = df[df["Store"] == pick].sort_values("Date")
    st.plotly_chart(px.line(sd, x="Date", y="Weekly_Sales",
                            title=f"Store {pick} — weekly sales"),
                    use_container_width=True)
