import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Stunnerz Spend & Sales Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path("stunnerz_skateboards_simulated_data.csv")
FALLBACK_DATA_PATH = Path("stunnerz_skateboards_simulated_data.csv")
DATE_COL = "date"
SALES_COL = "total_sales"
PROMO_COL = "promo"
WEEKDAY_COL = "weekday"

SPEND_COLUMNS = [
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

CHANNEL_LABELS = {
    "google_display": "Google Display",
    "google_search_brand": "Google Search Brand",
    "google_search_nob": "Google Search Non-Brand",
    "facebook_pr": "Facebook Prospecting",
    "facebook_rt": "Facebook Retargeting",
    "google_discovery": "Google Discovery",
    "bing_search_brand": "Bing Search Brand",
    "bing_shopping_feed": "Bing Shopping Feed",
    "pinterest_viz": "Pinterest Video",
    "pinterest_pr": "Pinterest Prospecting",
    "pinterest_rt": "Pinterest Retargeting",
    "google_pmax": "Google PMax",
}


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes | None, file_name: str | None) -> pd.DataFrame:
    """Load and validate the dataset."""
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        path = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else FALLBACK_DATA_PATH
        #df = pd.read_csv(path)
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is None:
            st.markdown("### Upload your dataset to start the dashboard")
            st.stop()
        
        df = pd.read_csv(uploaded_file)

    expected_cols = {DATE_COL, SALES_COL, PROMO_COL, WEEKDAY_COL, *SPEND_COLUMNS}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError("Some date values could not be parsed. Check the date column format.")

    numeric_cols = [SALES_COL] + SPEND_COLUMNS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df[PROMO_COL] = df[PROMO_COL].astype(str).fillna("unknown")
    df[WEEKDAY_COL] = df[WEEKDAY_COL].astype(str).fillna("unknown")

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["total_spend"] = df[SPEND_COLUMNS].sum(axis=1)
    df["sales_per_dollar_spent"] = np.where(df["total_spend"] > 0, df[SALES_COL] / df["total_spend"], np.nan)
    df["active_channels"] = (df[SPEND_COLUMNS] > 0).sum(axis=1)
    return df


@st.cache_data(show_spinner=False)
def aggregate_data(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Aggregate the data to the chosen time grain."""
    freq_map = {
        "Daily": "D",
        "Weekly": "W-MON",
        "Monthly": "MS",
    }

    if grain == "Daily":
        agg = df.copy()
    else:
        freq = freq_map[grain]
        base = df.set_index(DATE_COL)
        agg_dict = {SALES_COL: "sum", "total_spend": "sum", "active_channels": "mean"}
        agg_dict.update({col: "sum" for col in SPEND_COLUMNS})
        agg = base.resample(freq).agg(agg_dict).reset_index()

        promo_mode = (
            base[PROMO_COL]
            .resample(freq)
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .reset_index(drop=True)
        )
        weekday_mode = (
            base[WEEKDAY_COL]
            .resample(freq)
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
            .reset_index(drop=True)
        )
        agg[PROMO_COL] = promo_mode
        agg[WEEKDAY_COL] = weekday_mode
        agg["sales_per_dollar_spent"] = np.where(agg["total_spend"] > 0, agg[SALES_COL] / agg["total_spend"], np.nan)

    agg["period_label"] = agg[DATE_COL].dt.strftime("%Y-%m-%d")
    return agg


@st.cache_data(show_spinner=False)
def make_long_spend(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=[DATE_COL, SALES_COL, "total_spend", PROMO_COL, WEEKDAY_COL],
        value_vars=SPEND_COLUMNS,
        var_name="channel",
        value_name="spend",
    )
    long_df["channel_label"] = long_df["channel"].map(CHANNEL_LABELS).fillna(long_df["channel"])
    long_df["spend_share"] = np.where(long_df["total_spend"] > 0, long_df["spend"] / long_df["total_spend"], 0.0)
    return long_df


@st.cache_data(show_spinner=False)
def build_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [SALES_COL, "total_spend"] + SPEND_COLUMNS
    corr = df[cols].corr(numeric_only=True)
    ranked = corr[SALES_COL].drop(SALES_COL).sort_values(ascending=False).rename("correlation_with_total_sales")
    result = ranked.reset_index().rename(columns={"index": "metric"})
    result["metric_label"] = result["metric"].map(CHANNEL_LABELS).fillna(result["metric"])
    return result[["metric", "metric_label", "correlation_with_total_sales"]]


@st.cache_data(show_spinner=False)
def pivot_summary(df: pd.DataFrame, group_field: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_field, dropna=False)
        .agg(
            periods=(DATE_COL, "count"),
            total_sales=(SALES_COL, "sum"),
            total_spend=("total_spend", "sum"),
            avg_sales=(SALES_COL, "mean"),
            avg_spend=("total_spend", "mean"),
        )
        .reset_index()
    )
    grouped["sales_per_dollar_spent"] = np.where(grouped["total_spend"] > 0, grouped["total_sales"] / grouped["total_spend"], np.nan)
    return grouped


st.title("📈 Stunnerz Spend & Sales Explorer")
st.caption(
    "Interactive Streamlit app for exploring sales outcomes against ad spend. "
    "Built for intuitive business exploration rather than fixed analysis."
)

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

    try:
        if uploaded_file is not None:
            raw_df = load_data_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
        else:
            raw_df = load_data_from_bytes(None, None)
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    min_date = raw_df[DATE_COL].min().date()
    max_date = raw_df[DATE_COL].max().date()

    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    grain = st.selectbox("Time grain", options=["Daily", "Weekly", "Monthly"], index=1)

    promo_values = sorted(raw_df[PROMO_COL].dropna().unique().tolist())
    selected_promos = st.multiselect("Promo type", options=promo_values, default=promo_values)

    weekday_values = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    present_weekdays = [day for day in weekday_values if day in raw_df[WEEKDAY_COL].unique()]
    selected_weekdays = st.multiselect("Weekdays", options=present_weekdays, default=present_weekdays)

    selected_channels = st.multiselect(
        "Channels to display",
        options=SPEND_COLUMNS,
        default=SPEND_COLUMNS[:6],
        format_func=lambda x: CHANNEL_LABELS.get(x, x),
    )

    hide_zero_sales = st.checkbox("Hide zero-sales rows", value=False)
    hide_zero_spend = st.checkbox("Hide zero-spend rows", value=False)

filtered_raw = raw_df.loc[
    (raw_df[DATE_COL].dt.date >= start_date)
    & (raw_df[DATE_COL].dt.date <= end_date)
    & (raw_df[PROMO_COL].isin(selected_promos))
    & (raw_df[WEEKDAY_COL].isin(selected_weekdays))
].copy()

if hide_zero_sales:
    filtered_raw = filtered_raw[filtered_raw[SALES_COL] > 0]
if hide_zero_spend:
    filtered_raw = filtered_raw[filtered_raw["total_spend"] > 0]

if filtered_raw.empty:
    st.warning("No rows match the selected filters. Adjust the sidebar controls.")
    st.stop()

view_df = aggregate_data(filtered_raw, grain)
long_spend = make_long_spend(view_df)

if not selected_channels:
    selected_channels = SPEND_COLUMNS[:4]

channel_view = long_spend[long_spend["channel"].isin(selected_channels)].copy()

# KPI row
period_count = len(view_df)
total_sales = float(view_df[SALES_COL].sum())
total_spend = float(view_df["total_spend"].sum())
ratio = total_sales / total_spend if total_spend > 0 else np.nan
avg_active_channels = float(view_df["active_channels"].mean()) if "active_channels" in view_df else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"${total_sales:,.0f}")
k2.metric("Total Spend", f"${total_spend:,.0f}")
k3.metric("Sales / Spend", f"{ratio:,.2f}" if pd.notna(ratio) else "N/A")
k4.metric("Periods in View", f"{period_count:,}")

st.divider()

col1, col2 = st.columns([1.6, 1.0])
with col1:
    st.subheader("Sales and spend over time")
    metric_options = [SALES_COL, "total_spend"] + selected_channels
    selected_metric_lines = st.multiselect(
        "Metrics to plot",
        options=metric_options,
        default=[SALES_COL, "total_spend"],
        format_func=lambda x: CHANNEL_LABELS.get(x, x.replace("_", " ").title()),
        key="metric_lines",
    )
    if not selected_metric_lines:
        selected_metric_lines = [SALES_COL, "total_spend"]

    ts_df = view_df[[DATE_COL] + selected_metric_lines].melt(id_vars=DATE_COL, var_name="metric", value_name="value")
    ts_df["metric_label"] = ts_df["metric"].map(CHANNEL_LABELS).fillna(ts_df["metric"].str.replace("_", " ").str.title())

    fig_ts = px.line(
        ts_df,
        x=DATE_COL,
        y="value",
        color="metric_label",
        markers=True,
        template="plotly_white",
    )
    fig_ts.update_layout(height=430, legend_title_text="Metric", yaxis_title="Value", xaxis_title=None)
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader("Spend mix snapshot")
    spend_by_channel = (
        channel_view.groupby("channel_label", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
    )
    fig_mix = px.bar(
        spend_by_channel,
        x="spend",
        y="channel_label",
        orientation="h",
        template="plotly_white",
        text_auto=".2s",
    )
    fig_mix.update_layout(height=430, xaxis_title="Spend", yaxis_title=None, showlegend=False)
    st.plotly_chart(fig_mix, use_container_width=True)

col3, col4 = st.columns([1.2, 1.4])
with col3:
    st.subheader("Spend vs sales explorer")
    scatter_channel = st.selectbox(
        "Choose a channel",
        options=selected_channels,
        format_func=lambda x: CHANNEL_LABELS.get(x, x),
        key="scatter_channel",
    )
    fig_scatter = px.scatter(
        view_df,
        x=scatter_channel,
        y=SALES_COL,
        size="total_spend",
        color=PROMO_COL if PROMO_COL in view_df.columns else None,
        hover_data={DATE_COL: True, "total_spend": ":,.2f", SALES_COL: ":,.2f"},
        template="plotly_white",
    )
    fig_scatter.update_layout(
        height=430,
        xaxis_title=CHANNEL_LABELS.get(scatter_channel, scatter_channel),
        yaxis_title="Total Sales",
        legend_title_text="Promo",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    st.subheader("Channel mix over time")
    mix_mode = st.radio("View", ["Absolute spend", "Spend share"], horizontal=True)
    area_df = channel_view.copy()
    value_col = "spend" if mix_mode == "Absolute spend" else "spend_share"
    fig_area = px.area(
        area_df,
        x=DATE_COL,
        y=value_col,
        color="channel_label",
        template="plotly_white",
    )
    fig_area.update_layout(
        height=430,
        xaxis_title=None,
        yaxis_title="Spend" if mix_mode == "Absolute spend" else "Share of spend",
        legend_title_text="Channel",
    )
    st.plotly_chart(fig_area, use_container_width=True)

st.divider()

left, right = st.columns([1.1, 1.3])
with left:
    st.subheader("Channel comparison table")
    summary_df = (
        channel_view.groupby(["channel", "channel_label"], as_index=False)
        .agg(
            total_spend=("spend", "sum"),
            avg_period_spend=("spend", "mean"),
            max_period_spend=("spend", "max"),
            active_periods=("spend", lambda s: int((s > 0).sum())),
        )
        .sort_values("total_spend", ascending=False)
    )
    summary_df["share_of_selected_spend"] = np.where(
        summary_df["total_spend"].sum() > 0,
        summary_df["total_spend"] / summary_df["total_spend"].sum(),
        0.0,
    )
    st.dataframe(
        summary_df[[
            "channel_label",
            "total_spend",
            "share_of_selected_spend",
            "avg_period_spend",
            "max_period_spend",
            "active_periods",
        ]],
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Correlation helper")
    corr_df = correlation_table(view_df)
    corr_df = corr_df[corr_df["metric"].isin(["total_spend", *selected_channels])].copy()
    fig_corr = px.bar(
        corr_df.sort_values("correlation_with_total_sales", ascending=True),
        x="correlation_with_total_sales",
        y="metric_label",
        orientation="h",
        template="plotly_white",
        text_auto=".2f",
    )
    fig_corr.update_layout(height=380, xaxis_title="Correlation with Total Sales", yaxis_title=None)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Use this as an exploration aid only. It helps users spot patterns but is not a causal analysis.")

st.divider()

st.subheader("Flexible business cuts")
pivot_field = st.selectbox(
    "Group the data by",
    options=[PROMO_COL, WEEKDAY_COL],
    format_func=lambda x: x.replace("_", " ").title(),
)
cut_df = pivot_summary(view_df, pivot_field)

cut_fig_metric = st.selectbox(
    "Metric to visualize",
    options=["total_sales", "total_spend", "avg_sales", "avg_spend", "sales_per_dollar_spent"],
    format_func=lambda x: x.replace("_", " ").title(),
)
fig_cut = px.bar(
    cut_df,
    x=pivot_field,
    y=cut_fig_metric,
    template="plotly_white",
    text_auto=".2s",
)
fig_cut.update_layout(height=380, xaxis_title=None, yaxis_title=cut_fig_metric.replace("_", " ").title())
st.plotly_chart(fig_cut, use_container_width=True)

with st.expander("See grouped table"):
    st.dataframe(cut_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Data explorer")
show_columns = st.multiselect(
    "Columns to show",
    options=view_df.columns.tolist(),
    default=[DATE_COL, SALES_COL, "total_spend", PROMO_COL, WEEKDAY_COL] + selected_channels[:3],
)
rows_to_show = st.slider("Rows to display", min_value=10, max_value=min(500, len(view_df)), value=min(100, len(view_df)), step=10)
if show_columns:
    st.dataframe(view_df[show_columns].head(rows_to_show), use_container_width=True, hide_index=True)
else:
    st.info("Select at least one column to show the data explorer.")

csv_bytes = build_download(view_df)
st.download_button(
    "Download filtered view as CSV",
    data=csv_bytes,
    file_name="stunnerz_filtered_view.csv",
    mime="text/csv",
)

st.divider()
with st.expander("How this app is designed"):
    st.markdown(
        """
        - It accepts the provided Stunnerz dataset out of the box and also supports uploading a replacement CSV.
        - It is built for interactive exploration, so account managers can filter, compare, and inspect the data without touching code.
        - It avoids heavy model logic and keeps the app responsive with cached data-loading and transformation steps.
        - It keeps analysis light on purpose: the app productionalizes access to the data so another stakeholder can do the actual interpretation.
        """
    )
