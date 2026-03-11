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

# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(
    page_title="Stunnerz Skateboards",
    page_icon="🛹",
    layout="wide"
)

# ---------------------------------
# Fixed configuration
# ---------------------------------
APP_TITLE = "Stunnerz Skateboards"
APP_SUBTITLE = "Marketing spend analytics dashboard"
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

# ---------------------------------
# Styling
# ---------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .section-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .small-label {
        font-size: 0.9rem;
        color: #9ca3af;
    }
    .big-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Helper functions
# ---------------------------------
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


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def safe_corr(s1: pd.Series, s2: pd.Series) -> float:
    temp = pd.concat([s1, s2], axis=1).dropna()
    if len(temp) < 2:
        return np.nan
    return temp.iloc[:, 0].corr(temp.iloc[:, 1])


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return clean_columns(df)


def make_metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">{label}</div>
            <div class="big-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_info_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="small-label">{title}</div>
            <div class="big-value" style="font-size:1.1rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------
# Header
# ---------------------------------
st.title(f"🛹 {APP_TITLE}")
st.caption(APP_SUBTITLE)

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.markdown("### Upload your dataset to start the dashboard")
    st.markdown(
        """
        This app expects these columns:

        - `date`
        - `total_sales`
        - ad spend columns
        - `promo`
        - `weekday`
        """
    )
    st.stop()

# ---------------------------------
# Load and validate data
# ---------------------------------
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

required_cols = ["date", OUTCOME_COL] + CONTROL_COLS + SPEND_COLS
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error("Your CSV is missing required columns.")
    st.write("Missing columns:")
    st.write(missing_cols)
    st.write("Columns found in your file:")
    st.write(list(df.columns))
    st.stop()

# ---------------------------------
# Type cleaning
# ---------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

for col in [OUTCOME_COL] + CONTROL_COLS + SPEND_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[OUTCOME_COL]).reset_index(drop=True)

if df.empty:
    st.error("The dataset is empty after cleaning.")
    st.stop()

# ---------------------------------
# Filters
# ---------------------------------
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

rolling_window = st.sidebar.slider("Rolling average window", 2, 12, 4)
corr_window = st.sidebar.slider("Rolling correlation window", 2, 12, 4)

selected_channel = st.sidebar.selectbox(
    "Select channel",
    SPEND_COLS
)

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

# ---------------------------------
# Derived fields
# ---------------------------------
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

channel_totals = df[SPEND_COLS].sum().sort_values(ascending=False)
channel_corrs = pd.Series(
    {channel: safe_corr(df[channel], df[OUTCOME_COL]) for channel in SPEND_COLS}
).sort_values(ascending=False)

top_sales_idx = df[OUTCOME_COL].idxmax()
top_sales_date = df.loc[top_sales_idx, "date"].date()

# ---------------------------------
# KPI section
# ---------------------------------
total_sales = df[OUTCOME_COL].sum()
total_spend = df["total_spend"].sum()
avg_sales = df[OUTCOME_COL].mean()
avg_spend = df["total_spend"].mean()
spend_sales_ratio = total_spend / total_sales if total_sales != 0 else np.nan

m1, m2, m3, m4 = st.columns(4)
with m1:
    make_metric_card("Total Sales", format_currency(total_sales))
with m2:
    make_metric_card("Total Spend", format_currency(total_spend))
with m3:
    make_metric_card("Avg Weekly Sales", format_currency(avg_sales))
with m4:
    make_metric_card("Spend / Sales Ratio", "N/A" if pd.isna(spend_sales_ratio) else f"{spend_sales_ratio:.2f}")

# ---------------------------------
# Executive summary row
# ---------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    make_info_card("Highest Spend Channel", channel_totals.index[0])
with c2:
    best_corr = channel_corrs.dropna().index[0] if not channel_corrs.dropna().empty else "N/A"
    make_info_card("Most Correlated With Sales", best_corr)
with c3:
    make_info_card("Top Sales Date", str(top_sales_date))

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Executive Overview", "Spend Allocation", "Channel Analysis", "Diagnostics", "Data Preview"]
)

# ---------------------------------
# Tab 1
# ---------------------------------
with tab1:
    left, right = st.columns(2)

    with left:
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
            yaxis_title="Sales",
            legend_title=""
        )
        st.plotly_chart(fig_sales, use_container_width=True)

    with right:
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
            yaxis2=dict(title="Spend", overlaying="y", side="right"),
            legend_title=""
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    summary_df = pd.DataFrame({
        "Metric": ["Selected Channel", "Selected Channel Total Spend", "Selected Channel Correlation With Sales"],
        "Value": [
            selected_channel,
            format_currency(df[selected_channel].sum()),
            "N/A" if pd.isna(safe_corr(df[selected_channel], df[OUTCOME_COL])) else f"{safe_corr(df[selected_channel], df[OUTCOME_COL]):.2f}"
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------------------------------
# Tab 2
# ---------------------------------
with tab2:
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
            spend_share_df["total_spend"] / grand_total_spend if grand_total_spend != 0 else 0
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

# ---------------------------------
# Tab 3
# ---------------------------------
with tab3:
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

    channel_summary = pd.DataFrame({
        "channel": SPEND_COLS,
        "total_spend": [df[ch].sum() for ch in SPEND_COLS],
        "correlation_with_sales": [safe_corr(df[ch], df[OUTCOME_COL]) for ch in SPEND_COLS]
    }).sort_values("correlation_with_sales", ascending=False)

    st.subheader("Channel Summary")
    st.dataframe(channel_summary, use_container_width=True, hide_index=True)

# ---------------------------------
# Tab 4
# ---------------------------------
with tab4:
    left, right = st.columns(2)

    diag_cols = [OUTCOME_COL] + SPEND_COLS + CONTROL_COLS
    corr_matrix = df[diag_cols].corr(numeric_only=True)

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

# ---------------------------------
# Tab 5
# ---------------------------------
with tab5:
    st.subheader("Filtered Data Preview")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="stunnerz_skateboards_filtered_data.csv",
        mime="text/csv"
    )
