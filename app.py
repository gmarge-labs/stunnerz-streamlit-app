import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Stunnerz Skateboards",
    layout="wide"
)

# -----------------------------
# Fixed app settings
# -----------------------------
APP_TITLE = "Stunnerz Skateboards"
OUTCOME_COL = "total_sales"
CONTROL_COLS = ["promo", "weekday"]
SPEND_COLS = [
    "google_display",
    "google_search_brand",
    "google_search_nob",
    "facebook_pr",
    "facebook_rt",
    "google_discovery",
    "bing_search_brand",
    "bing_shopping_feed",
    "pinterest_viz",
    "pinterest_pr",
    "pinterest_rt",
    "google_pmax",
]

# -----------------------------
# Helpers
# -----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

def format_currency(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"${x:,.0f}"

def safe_corr(s1: pd.Series, s2: pd.Series) -> float:
    temp = pd.concat([s1, s2], axis=1).dropna()
    if len(temp) < 2:
        return np.nan
    return temp.iloc[:, 0].corr(temp.iloc[:, 1])

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = clean_columns(df)
    return df

# -----------------------------
# Header
# -----------------------------
st.title(APP_TITLE)
st.caption("Interactive spend analytics dashboard")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.markdown("### Upload your dataset to start the dashboard")
    st.stop()

# -----------------------------
# Load data
# -----------------------------
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

# -----------------------------
# Validate required columns
# -----------------------------
required_cols = ["date", OUTCOME_COL] + CONTROL_COLS + SPEND_COLS
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error("Your CSV is missing required columns.")
    st.write("Missing columns:")
    st.write(missing_cols)
    st.write("Columns found in your file:")
    st.write(list(df.columns))
    st.stop()

# -----------------------------
# Clean types
# -----------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

for col in [OUTCOME_COL] + CONTROL_COLS + SPEND_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where total_sales is missing
df = df.dropna(subset=[OUTCOME_COL]).reset_index(drop=True)

if df.empty:
    st.error("The dataset is empty after cleaning. Check your file.")
    st.stop()

# -----------------------------
# Sidebar controls after load
# -----------------------------
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

rolling_window = st.sidebar.slider(
    "Rolling average window",
    min_value=2,
    max_value=12,
    value=4
)

corr_window = st.sidebar.slider(
    "Rolling correlation window",
    min_value=2,
    max_value=12,
    value=4
)

selected_channel = st.sidebar.selectbox(
    "Select channel",
    options=SPEND_COLS
)

# -----------------------------
# Date filter
# -----------------------------
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

df = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

if df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

# -----------------------------
# Derived fields
# -----------------------------
df["total_spend"] = df[SPEND_COLS].fillna(0).sum(axis=1)
df["sales_rolling"] = df[OUTCOME_COL].rolling(rolling_window).mean()
df["spend_rolling"] = df["total_spend"].rolling(rolling_window).mean()
df["rolling_corr"] = df[selected_channel].rolling(corr_window).corr(df[OUTCOME_COL])

long_spend = df.melt(
    id_vars=["date"],
    value_vars=SPEND_COLS,
    var_name="channel",
    value_name="spend"
)

# -----------------------------
# KPI metrics
# -----------------------------
total_sales = df[OUTCOME_COL].sum()
total_spend = df["total_spend"].sum()
avg_sales = df[OUTCOME_COL].mean()
avg_spend = df["total_spend"].mean()
spend_sales_ratio = total_spend / total_sales if total_sales != 0 else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", format_currency(total_sales))
k2.metric("Total Spend", format_currency(total_spend))
k3.metric("Avg Weekly Sales", format_currency(avg_sales))
k4.metric("Spend / Sales Ratio", f"{spend_sales_ratio:.2f}" if pd.notna(spend_sales_ratio) else "N/A")

# -----------------------------
# Quick insights
# -----------------------------
channel_totals = df[SPEND_COLS].sum().sort_values(ascending=False)
channel_corrs = pd.Series(
    {channel: safe_corr(df[channel], df[OUTCOME_COL]) for channel in SPEND_COLS}
).sort_values(ascending=False)

top_sales_idx = df[OUTCOME_COL].idxmax()

with st.expander("Quick business insights", expanded=True):
    c1, c2 = st.columns(2)
    c1.write(f"**Highest spend channel:** {channel_totals.index[0]}")
    c1.write(f"**Lowest spend channel:** {channel_totals.index[-1]}")
    c2.write(
        f"**Most correlated with sales:** "
        f"{channel_corrs.dropna().index[0] if not channel_corrs.dropna().empty else 'N/A'}"
    )
    c2.write(f"**Top sales date:** {df.loc[top_sales_idx, 'date'].date()}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Spend Allocation", "Channel Analysis", "Diagnostics", "Data Preview"]
)

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab1:
    st.subheader("Sales trend")

    fig_sales = go.Figure()
    fig_sales.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[OUTCOME_COL],
            mode="lines",
            name="Total Sales"
        )
    )
    fig_sales.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sales_rolling"],
            mode="lines",
            name=f"{rolling_window}-period rolling average"
        )
    )
    fig_sales.update_layout(
        title="Sales Trend",
        height=420,
        xaxis_title="Date",
        yaxis_title="Sales"
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    fig_dual = go.Figure()
    fig_dual.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[OUTCOME_COL],
            mode="lines",
            name="Total Sales",
            yaxis="y1"
        )
    )
    fig_dual.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["total_spend"],
            mode="lines",
            name="Total Spend",
            yaxis="y2"
        )
    )
    fig_dual.update_layout(
        title="Sales vs Total Spend",
        height=420,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Spend", overlaying="y", side="right")
    )
    st.plotly_chart(fig_dual, use_container_width=True)

# -----------------------------
# Tab 2: Spend Allocation
# -----------------------------
with tab2:
    st.subheader("Budget allocation")

    left, right = st.columns(2)

    with left:
        spend_totals_df = channel_totals.reset_index()
        spend_totals_df.columns = ["channel", "total_spend"]

        fig_bar = px.bar(
            spend_totals_df,
            x="total_spend",
            y="channel",
            orientation="h",
            title="Total Spend by Channel"
        )
        fig_bar.update_layout(
            height=420,
            xaxis_title="Spend",
            yaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        spend_share_df = spend_totals_df.copy()
        grand_total_spend = spend_share_df["total_spend"].sum()
        spend_share_df["spend_share"] = (
            spend_share_df["total_spend"] / grand_total_spend
            if grand_total_spend != 0 else 0
        )

        fig_share = px.bar(
            spend_share_df.sort_values("spend_share", ascending=False),
            x="channel",
            y="spend_share",
            title="Spend Share by Channel"
        )
        fig_share.update_layout(
            height=420,
            xaxis_title="Channel",
            yaxis_title="Share"
        )
        st.plotly_chart(fig_share, use_container_width=True)

    fig_area = px.area(
        long_spend,
        x="date",
        y="spend",
        color="channel",
        title="Weekly Spend by Channel"
    )
    fig_area.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Spend"
    )
    st.plotly_chart(fig_area, use_container_width=True)

# -----------------------------
# Tab 3: Channel Analysis
# -----------------------------
with tab3:
    st.subheader("Selected channel vs sales")

    left, right = st.columns(2)

    with left:
        fig_scatter = px.scatter(
            df,
            x=selected_channel,
            y=OUTCOME_COL,
            trendline="ols",
            title=f"{selected_channel} vs Total Sales"
        )
        fig_scatter.update_layout(
            height=420,
            xaxis_title=selected_channel,
            yaxis_title="Total Sales"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right:
        fig_corr = px.line(
            df,
            x="date",
            y="rolling_corr",
            title=f"Rolling Correlation: {selected_channel} vs Total Sales"
        )
        fig_corr.add_hline(y=0, line_dash="dash")
        fig_corr.update_layout(
            height=420,
            xaxis_title="Date",
            yaxis_title="Rolling Correlation"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    corr_table = pd.DataFrame({
        "channel": SPEND_COLS,
        "correlation_with_sales": [channel_corrs.get(ch, np.nan) for ch in SPEND_COLS],
        "total_spend": [df[ch].sum() for ch in SPEND_COLS]
    }).sort_values("correlation_with_sales", ascending=False)

    st.subheader("Channel summary")
    st.dataframe(corr_table, use_container_width=True)

# -----------------------------
# Tab 4: Diagnostics
# -----------------------------
with tab4:
    st.subheader("Diagnostics")

    diag_cols = [OUTCOME_COL] + SPEND_COLS + CONTROL_COLS
    corr_matrix = df[diag_cols].corr(numeric_only=True)

    left, right = st.columns(2)

    with left:
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

    with right:
        fig_hist = px.histogram(
            df,
            x=OUTCOME_COL,
            nbins=25,
            title="Distribution of Total Sales"
        )
        fig_hist.update_layout(height=290)
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(
            df,
            y=OUTCOME_COL,
            title="Total Sales Outliers"
        )
        fig_box.update_layout(height=290)
        st.plotly_chart(fig_box, use_container_width=True)

    zero_spend_df = pd.DataFrame({
        "channel": SPEND_COLS,
        "zero_spend_pct": [(df[col].fillna(0) == 0).mean() for col in SPEND_COLS]
    }).sort_values("zero_spend_pct", ascending=False)

    fig_zero = px.bar(
        zero_spend_df,
        x="channel",
        y="zero_spend_pct",
        title="Percent of Periods with Zero Spend"
    )
    fig_zero.update_layout(
        height=380,
        xaxis_title="Channel",
        yaxis_title="Zero Spend %"
    )
    st.plotly_chart(fig_zero, use_container_width=True)

# -----------------------------
# Tab 5: Data Preview
# -----------------------------
with tab5:
    st.subheader("Filtered data preview")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="stunnerz_skateboards_filtered_data.csv",
        mime="text/csv"
    )
