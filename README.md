# London Crime Pulse Explorer

Interactive exploration of real London street-level crime data, built toward a 3D hexbin map with monthly timelapse behaviour.

## Data source

Crime records come from [data.police.uk](https://data.police.uk/) (street-level crime open data).

**Methodology note:** Published coordinates are anonymised and snapped to representative points on streets. They indicate the general area of a reported crime, not the exact location.

## Testing strategy

Development and testing start with **May 2025** (`2025-05`) only. Once the pipeline works end-to-end for that month, additional months are added incrementally.

## Project structure

```
data/raw/          # Untouched monthly CSV files (local; not committed)
data/processed/    # Cleaned row-level parquet
data/viz/          # Aggregation / map-ready datasets
data/reference/    # Optional reference files
src/               # Ingestion, cleaning, validation, visualization
outputs/           # Reports, charts, exported maps
app.py             # Streamlit entry point
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Status

Mini 0 — project skeleton only. No data loader, cleaning pipeline, or maps yet.
