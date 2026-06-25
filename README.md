# London Crime Explorer

Interactive 3D map of real London street-level crime data. A batch pipeline turns monthly police CSV files into a compact hex aggregation; a Streamlit app lets you explore crime intensity by month and crime type on a tilted PyDeck hex column map.

## Data source

Crime records come from [data.police.uk](https://data.police.uk/) (street-level crime open data).

**Methodology note:** Published coordinates are anonymised and snapped to representative points on streets. They indicate the general area of a reported crime, not the exact location.

## What works today

- **Ingestion** — load `*-street.csv` files from `data/raw/<YYYY-MM>/`
- **Cleaning** — standardise columns, validate coordinates, filter to `AREA_BBOX`
- **Aggregation** — bin crimes into a hex grid (month × crime type × cell)
- **App** — Streamlit UI with month slider, crime-type pills, summary metrics, and a 3D map

The app reads one precomputed file (`data/viz/crime_hex_3d_month.parquet`). It does not load raw CSVs at runtime.

## Project structure

```
data/raw/              # Monthly CSV folders: data/raw/2025-05/*-street.csv (local; not committed)
data/processed/        # Optional cleaned row-level Parquet (pipeline side output)
data/viz/              # Map-ready aggregation (crime_hex_3d_month.parquet)
src/
  config.py            # Paths, pipeline months, AREA_BBOX, hex/map settings
  ingestion/           # load_crime_files.py — read raw CSVs
  cleaning/            # clean_crime_data.py — standardise and validate rows
  transformation/      # hex_grid.py, aggregate_hex_grid.py — hex binning + Parquet output
  visualization/       # hexagon_3d_map.py — colour, elevation, PyDeck rendering
run_pipeline.py        # Batch pipeline: raw CSVs → viz Parquet
app.py                 # Streamlit entry point
```

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Add raw data

Download street-level crime CSVs from [data.police.uk](https://data.police.uk/) and place them under month folders:

```
data/raw/2025-05/
  2025-05-metropolitan-street.csv
  2025-05-city-of-london-street.csv
  ...
```

Only `*-street.csv` files are used; other files in the folder are ignored.

### 3. Run the pipeline

```bash
python run_pipeline.py
```

This writes `data/viz/crime_hex_3d_month.parquet`. Re-run whenever raw data or pipeline settings change.

### 4. Launch the app

```bash
streamlit run app.py
```

## Configuration

Edit `src/config.py`:

| Setting | Purpose |
|---------|---------|
| `PIPELINE_MONTHS` | Months to process. `["2025-05"]` for one month, `None` for every `YYYY-MM` folder under `data/raw/`. |
| `DEFAULT_TESTING_MONTH` | Initial month in the app slider (defaults to `2025-05` when processing all months). |
| `AREA_BBOX` | Longitude/latitude limits for filtering crime points and centring the map. Defaults to Greater London. **If your data is from outside London, update these values** to match your study area, then re-run the pipeline. |
| `HEX_RADIUS_METERS` | Hex cell size — change requires re-running the pipeline. |
| `HEX_ELEVATION_*`, `HEX_COLOR_*`, `HEX_CAMERA_*`, `HEX_MAP_HEIGHT` | Map appearance — app restart is enough (no pipeline re-run unless bin size changes). |

## Pipeline overview

```
data/raw/YYYY-MM/*-street.csv
  → load + clean (ingestion + cleaning)
  → hex aggregate (transformation)
  → data/viz/crime_hex_3d_month.parquet
  → Streamlit app + 3D map (visualization)
```

`run_pipeline.py` orchestrates the full batch path. Individual stages can also be run in isolation:

```bash
python -m src.ingestion.load_crime_files
python -m src.cleaning.clean_crime_data
```

## Requirements

- Python 3.10+
- See `requirements.txt`: streamlit, pandas, pyarrow, pydeck
