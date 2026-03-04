# TS Anomaly Tool

Streamlit app for time series forecasting and anomaly detection from one or more CSV files.

<p align="center">
  <img src="assets/forecasting.png" width="48%" />
  <img src="assets/anomaly.png" width="48%" />
</p>


## Live demo (temporary link)

Current public demo (Cloudflare quick tunnel):

https://peninsula-purchased-parallel-miss.trycloudflare.com/

Note: this is a quick tunnel URL and may change if the tunnel restarts.

## Parameters (what they mean)

- **Forecast steps**  
  Number of future time points to forecast (default: 12). Used for the forecast plot.

- **Forecast frequency** (e.g., `2D`)  
  The resampling interval for the unified time grid. `2D` means every 2 days.  
  This also controls the spacing of forecast timestamps.

- **Max lag**  
  Maximum lag (±L) used when computing cross-correlation (CCF) between variables to select correlated variables for multivariate modeling.

- **Correlation threshold**  
  Minimum absolute CCF value required to include a variable as “correlated” with the target.  
  Higher values select fewer variables and make the multivariate model more conservative.

- **Interpolation method**  
  How missing values are filled after resampling.  
  - `pchip` (default): shape-preserving interpolation (helps avoid overshoot), then clipped to be non-negative  
  - `polynomial`: polynomial interpolation (order controlled below), then clipped to be non-negative  
  - `linear`: linear interpolation, then clipped to be non-negative

- **Polynomial order**  
  Only used when interpolation method is `polynomial`.

  ## Outputs

After each run the app generates:

### `summary.csv`
One row per target variable (compound) with:
- **Compound**: the target column name (prefixed by category)
- **Model**: `SARIMAX` for univariate case, `VAR` for multivariate case
- **Correlated_Variables**: variables selected via CCF for the target (includes the target itself)
- **Correlation_Details**: correlation strength and lag for each selected variable
- **Lag_Order / Selected_Lag_Order**: lag used in the model (when applicable)
- **Anomaly_Count**: number of detected anomalies (observed-only)
- **Notes**: any warnings (e.g., insufficient data)

### `anomalies.txt`
A human-readable report listing anomaly events:
- **Isolated Deviation**: the target residual exceeds the anomaly threshold (2σ), with no clear correlation trend violation
- **Correlation Trend Violation** (VAR only): residual anomaly coincides with a mismatch in expected directionality between target and a correlated variable (based on lag-aware correlation)
- **No Correlated Data**: anomaly detected but correlated variables were interpolated/missing at that timestamp

### Figures
Saved under `figures/`:
- `*_forecast.png`: forecast plot
- `*_anomalies.png`: time series plot with anomaly points highlighted

### `outputs.zip`
A bundle containing `summary.csv`, `anomalies.txt`, and all figures.

### Example inputs

Two small example CSVs are included:

- examples/Antidepressants.csv

- examples/Antibiotics.csv

Upload one or both of them in the app. Use category names like Antidepressants and Antibiotics so your merged columns become Antidepressants_Drugs, Antibiotics_Drugs.

## Docker (easy server deployment)

This project is dockerized. You can build and run the app locally or on a server:

```bash
sudo docker build -t timeseries-tool:latest .
sudo docker rm -f timeseries-tool 2>/dev/null || true
sudo docker run -d --restart unless-stopped --name timeseries-tool -p 8501:8501 timeseries-tool:latest


Upload one or more CSV files with:
- `Date` column (parseable datetime)
- one or more numeric columns (example: `Drugs`)



The app runs forecasting + anomaly detection and produces:
- `summary.csv`
- `anomalies.txt`
- plots (PNGs)
- downloadable `outputs.zip`




## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[app]"
streamlit run streamlit_app.py
```
## Important note about the demo link
Since it’s a quick tunnel, it can change. If you want a stable URL in README, we should switch to a **named Cloudflare tunnel** + domain (or at least avoid hardcoding the link and instead say “Ask me for the current demo link”).
