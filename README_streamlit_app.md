# Stunnerz Spend & Sales Explorer

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — deployment dependencies
- `.streamlit/config.toml` — minimal Streamlit config
- `stunnerz_skateboards_simulated_data.csv` — dataset expected by the app

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the CSV file to a GitHub repo.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Set the entry point to `app.py`.
4. Deploy.

## What the app does
- Interactive date, promo, weekday, and channel filtering
- Daily, weekly, and monthly views
- Time-series charts for sales, spend, and channel trends
- Spend mix charts and channel comparison tables
- Scatter explorer for spend vs sales by channel
- Grouped business cuts by promo or weekday
- Filtered data export

## Notes
This app is intentionally exploration-first rather than analysis-first. It helps business users inspect the data intuitively so they can ask better questions before deeper modeling work starts.
