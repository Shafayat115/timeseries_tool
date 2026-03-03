from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, IO, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from orbit.models import DLT
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ---------------------------
# Config
# ---------------------------
@dataclass(frozen=True)
class PipelineConfig:
    forecast_steps: int = 12
    forecast_freq: str = "2D"
    max_lag: int = 2
    corr_threshold: float = 0.6
    interpolation_method: str = "pchip"
    interpolation_order: int = 2


# ---------------------------
# IO + helpers
# ---------------------------
def load_and_merge_data(files: List[Tuple[Union[str, IO[bytes], Any], str]]) -> pd.DataFrame:
    """
    files: [(file_like_or_path, category_name), ...]
    Each CSV must contain a Date column plus one or more numeric columns.
    """
    dfs = []
    for fileobj, category in files:
        df = pd.read_csv(fileobj, parse_dates=["Date"], engine="python")
        if "Date" not in df.columns:
            raise ValueError(f"{category}: Missing required column 'Date'")

        df = df.set_index("Date")
        df.index = df.index.ceil("D")
        df = df.sort_index()

        # coerce all value columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(how="all")

        # prefix
        df = df.rename(columns=lambda x: f"{category}_{x}")
        dfs.append(df)

    df_merged = pd.concat(dfs, axis=1, join="outer").sort_index()
    return df_merged


def resample_and_interpolate(df_merged: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date = df_merged.index.min()
    end_date = df_merged.index.max()
    new_index = pd.date_range(start=start_date, end=end_date, freq=cfg.forecast_freq)

    def round_to_nearest_index(date, index):
        return min(index, key=lambda x: abs(x - date))

    df_rounded = df_merged.reset_index()
    df_rounded["Date"] = df_rounded["Date"].apply(lambda x: round_to_nearest_index(x, new_index))
    df_rounded = df_rounded.groupby("Date").mean()
    df_raw = df_rounded.reindex(new_index)

    method = cfg.interpolation_method
    order = cfg.interpolation_order
    if method == "pchip":
        df_interpolated = df_raw.interpolate(method="pchip").clip(lower=0)
    elif method == "polynomial":
        df_interpolated = df_raw.interpolate(method="polynomial", order=order).clip(lower=0)
    else:
        df_interpolated = df_raw.interpolate().clip(lower=0)

    return df_raw, df_interpolated


def max_ccf(series1: pd.Series, series2: pd.Series, max_lag: int = 5) -> tuple[float, int]:
    lags = range(-max_lag, max_lag + 1)
    max_corr = 0.0
    best_lag = 0
    for lag in lags:
        if lag > 0:
            corr = series1.corr(series2.shift(lag))
        elif lag < 0:
            corr = series1.shift(-lag).corr(series2)
        else:
            corr = series1.corr(series2)

        if pd.notnull(corr) and abs(corr) > abs(max_corr):
            max_corr = float(corr)
            best_lag = int(lag)
    return max_corr, best_lag


def set_x_ticks(ax: plt.Axes, idx: pd.DatetimeIndex, freq: str) -> None:
    xticks = pd.date_range(start=idx.min(), end=idx.max(), freq=freq)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", rotation=90, labelsize=8)


def init_anomaly_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Anomaly Report\n")
        f.write("=" * 50 + "\n\n")


def update_anomaly_report(
    anomaly_log_path: Path,
    df_subset: pd.DataFrame,
    observed_residuals: pd.Series,
    obs_mask_subset: pd.DataFrame,
    subset_vars: list[str],
    target: str,
    corr_threshold: float,
    max_lag: int,
    model: str = "VAR",
) -> None:
    resid_std = observed_residuals.std()
    threshold = 2 * resid_std
    anomaly_dates = observed_residuals.index[observed_residuals.abs() > threshold]

    with open(anomaly_log_path, "a", encoding="utf-8") as log_file:
        for date in anomaly_dates:
            target_resid = float(observed_residuals.loc[date])

            if model == "SARIMAX" or len(subset_vars) == 1:
                log_file.write(f"{target} | {date.date()} | SARIMAX | Isolated Deviation\n")
                log_file.write(f"  ↳ Residual {target_resid:.2f} exceeds 2σ = {threshold:.2f}\n")
                continue

            trend_breaks = []
            valid_corr_data = False

            for var in subset_vars:
                if var == target:
                    continue

                cc, lag = max_ccf(df_subset[target], df_subset[var], max_lag=max_lag)
                aligned_var = df_subset[var].shift(-lag)

                if date not in aligned_var.index or pd.isna(aligned_var.loc[date]):
                    continue

                # only compare to correlated vars if the var at that date is observed (not interpolated)
                if not bool(obs_mask_subset[var].get(date, False)):
                    continue

                valid_corr_data = True

                idx = df_subset.index.get_loc(date)
                if idx == 0:
                    continue
                prev = df_subset.index[idx - 1]
                delta_t = df_subset[target].loc[date] - df_subset[target].loc[prev]
                delta_v = aligned_var.loc[date] - aligned_var.loc[prev]

                if (cc > corr_threshold and np.sign(delta_t) != np.sign(delta_v)) or \
                   (cc < -corr_threshold and np.sign(delta_t) == np.sign(delta_v)):
                    dir_desc = f"{'+' if delta_v > 0 else '-'} in {var} → {'+' if delta_t > 0 else '-'} in {target}"
                    mismatch_note = f"{var} (r={cc:.2f}, lag={lag}): mismatch ({dir_desc})"
                    trend_breaks.append(mismatch_note)

            if trend_breaks:
                log_file.write(f"{target} | {date.date()} | VAR | Correlation Trend Violation\n")
                for tb in trend_breaks:
                    log_file.write(f"  ↳ {tb}\n")
            elif valid_corr_data:
                log_file.write(f"{target} | {date.date()} | VAR | Isolated Deviation\n")
                log_file.write(f"  ↳ Residual {target_resid:.2f} exceeds 2σ = {threshold:.2f}\n")
            else:
                log_file.write(f"{target} | {date.date()} | VAR | No Correlated Data\n")
                log_file.write(f"  ↳ Residual {target_resid:.2f} exceeds 2σ = {threshold:.2f}\n")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


# ---------------------------
# Main callable
# ---------------------------
def run_pipeline(
    files: List[Tuple[Union[str, IO[bytes], Any], str]],
    output_dir: Union[str, Path],
    cfg: PipelineConfig = PipelineConfig(),
) -> dict:
    """
    Runs your full pipeline and writes:
      - summary.csv
      - anomalies.txt
      - many PNG plots under output_dir/figures/

    Returns a dict with paths.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    anomaly_log_path = output_dir / "anomalies.txt"
    init_anomaly_report(anomaly_log_path)

    df_merged = load_and_merge_data(files)
    df_raw, df_interpolated = resample_and_interpolate(df_merged, cfg)

    n_obs = len(df_interpolated)
    n_eq = df_interpolated.shape[1]
    if n_obs <= n_eq + 1:
        raise ValueError("Not enough observations. Increase dataset size or use a less frequent resampling.")

    observed_mask = ~df_raw.isna()
    interpolated_mask = df_raw.isna()

    summary_results = []

    for target in df_interpolated.columns:
        subset_vars = [target]
        correlation_info = []

        for var in df_interpolated.columns:
            if var == target:
                continue
            cc, lag = max_ccf(
                df_interpolated[target].dropna(),
                df_interpolated[var].dropna(),
                max_lag=cfg.max_lag,
            )
            if abs(cc) >= cfg.corr_threshold:
                subset_vars.append(var)
                correlation_info.append(f"{target}~{var}: r={cc:.2f}, lag={lag}")

        df_subset = df_interpolated[subset_vars]
        obs_mask_subset = observed_mask[subset_vars]
        interp_mask_subset = interpolated_mask[subset_vars]

        model_type = None
        lag_order_used = None
        anomaly_count = None
        notes = ""
        selected_lag_order = "N/A"

        # ---------- Univariate SARIMAX ----------
        if len(subset_vars) == 1:
            model_type = "SARIMAX"
            notes = "No correlated variables above threshold; univariate SARIMAX used."

            target_series = df_raw[target]
            ts_clean = target_series.replace([np.inf, -np.inf], np.nan).dropna()
            if not (ts_clean.empty or ts_clean.nunique() <= 1):
                _ = adfuller(ts_clean)

            # debug series + ACF/PACF
            if len(ts_clean) >= 3 and ts_clean.nunique() > 1:
                fig, ax = plt.subplots(figsize=(20, 8))
                ax.plot(target_series, marker="o")
                ax.set_title(f"Quick Check Plot for {target}")
                set_x_ticks(ax, df_interpolated.index, cfg.forecast_freq)
                ax.grid(True, linestyle="--", alpha=0.5)
                _savefig(figures_dir / f"{target}_series_debug.png")

                n_lags_plot = min(20, int((len(ts_clean) - 1) * 0.5))

                plt.figure(figsize=(20, 8), dpi=250)
                plot_acf(ts_clean, lags=n_lags_plot)
                plt.title(f"ACF for {target}")
                _savefig(figures_dir / f"{target}_acf.png")

                plt.figure(figsize=(20, 8), dpi=250)
                plot_pacf(ts_clean, lags=n_lags_plot)
                plt.title(f"PACF for {target}")
                _savefig(figures_dir / f"{target}_pacf.png")

            # SARIMAX fit
            try:
                model_uni = SARIMAX(
                    target_series,
                    order=(cfg.max_lag, 1, cfg.max_lag),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model_uni_fit = model_uni.fit(disp=False)
            except Exception as e:
                notes += f" SARIMAX model failed: {e}"
                continue

            # BSTS forecast plot (your existing approach)
            ts_bsts = df_interpolated[[target]].reset_index()
            ts_bsts.columns = ["date", "value"]

            bsts_model = DLT(response_col="value", date_col="date", seasonality=12, seed=42)
            bsts_model.fit(ts_bsts)

            bsts_fit = bsts_model.predict(ts_bsts)
            ts_bsts["bsts_pred"] = bsts_fit["prediction"]

            future_all = bsts_model.predict(df=ts_bsts[["date", "value"]], forecast_horizon=cfg.forecast_steps)
            future_dates = pd.date_range(
                start=ts_bsts["date"].max(),
                periods=cfg.forecast_steps + 1,
                freq=cfg.forecast_freq,
            )[1:]

            if "forecast_index" in future_all.columns:
                future = (
                    future_all.groupby("forecast_index")["prediction"]
                    .mean()
                    .reset_index(drop=True)
                    .to_frame("prediction")
                )
            else:
                draws_per_step = max(1, len(future_all) // cfg.forecast_steps)
                tail = future_all.tail(draws_per_step * cfg.forecast_steps).reset_index(drop=True)
                future = (
                    tail.groupby(np.arange(len(tail)) // draws_per_step)["prediction"]
                    .mean()
                    .reset_index(drop=True)
                    .to_frame("prediction")
                )

            future["date"] = future_dates[: len(future)]
            future = future.sort_values("date").reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(ts_bsts["date"], ts_bsts["value"], linewidth=1.2, label="Series")
            ax.scatter(df_subset.index[obs_mask_subset[target]], df_subset[target][obs_mask_subset[target]],
                       s=30, edgecolor="black", linewidth=0.5, alpha=0.9, label="Observed")
            ax.scatter(df_subset.index[interp_mask_subset[target]], df_subset[target][interp_mask_subset[target]],
                       s=30, alpha=0.9, label="Interpolated")
            ax.plot(ts_bsts["date"], ts_bsts["bsts_pred"], linewidth=1.2, label="BSTS Fitted")
            ax.plot(future["date"], future["prediction"], linestyle="--", linewidth=2, label="BSTS Forecast")
            ax.set_title(f"BSTS Forecast for '{target}'")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            set_x_ticks(ax, pd.DatetimeIndex(list(df_subset.index) + list(future["date"])), cfg.forecast_freq)
            _savefig(figures_dir / f"{target}_bsts_forecast.png")

            # residual anomalies from SARIMAX
            fitted_values_uni = model_uni_fit.fittedvalues
            residuals_uni = df_subset[target].iloc[len(df_subset[target]) - len(fitted_values_uni):] - fitted_values_uni
            threshold = 2 * residuals_uni.std()
            residuals_obs_mask = obs_mask_subset[target].loc[residuals_uni.index]
            observed_residuals = residuals_uni[residuals_obs_mask]
            anomalies = observed_residuals.abs() > threshold
            anomaly_count = int(anomalies.sum())
            anomaly_indices = observed_residuals.index[anomalies]

            update_anomaly_report(
                anomaly_log_path=anomaly_log_path,
                df_subset=df_subset,
                observed_residuals=observed_residuals,
                obs_mask_subset=obs_mask_subset,
                subset_vars=[target],
                target=target,
                corr_threshold=cfg.corr_threshold,
                max_lag=cfg.max_lag,
                model="SARIMAX",
            )

            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(residuals_uni.index, residuals_uni, marker="o", label="Residuals")
            ax.hlines([threshold, -threshold], xmin=residuals_uni.index[0], xmax=residuals_uni.index[-1],
                      linestyles="--", label="Threshold")
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, observed_residuals.loc[anomaly_indices],
                           s=50, edgecolor="black", linewidth=0.5, label="Anomaly", zorder=5)
            ax.set_title(f"Residuals and Anomalies for '{target}' (SARIMAX)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            set_x_ticks(ax, residuals_uni.index, cfg.forecast_freq)
            _savefig(figures_dir / f"{target}_residuals_anomalies_univariate.png")

        # ---------- VAR ----------
        else:
            model_type = "VAR"

            n_obs_subset = len(df_subset)
            n_eq_subset = df_subset.shape[1]
            if n_obs_subset <= n_eq_subset + 1:
                notes = f"Not enough observations to fit a VAR model for variables {subset_vars}."
                continue

            df_train_var = df_subset[obs_mask_subset[target]]
            var_model = VAR(df_train_var)

            selected_lag = max(1, cfg.max_lag)
            selected_lag_order = selected_lag
            lag_order_used = selected_lag

            if len(df_subset) <= selected_lag:
                notes = f"Insufficient data for lag={selected_lag}."
                continue

            try:
                var_model_fit = var_model.fit(selected_lag)
            except Exception as e:
                notes += f" VAR model fitting failed: {e}"
                continue

            # residual anomalies from VAR
            fitted_values_var = var_model_fit.fittedvalues
            residuals_var = df_subset.iloc[selected_lag:][target] - fitted_values_var[target]
            threshold = 2 * residuals_var.std()
            residuals_obs_mask = obs_mask_subset[target].loc[residuals_var.index]
            observed_residuals_var = residuals_var[residuals_obs_mask]
            anomalies = observed_residuals_var.abs() > threshold
            anomaly_count = int(anomalies.sum())
            anomaly_indices = observed_residuals_var.index[anomalies]

            update_anomaly_report(
                anomaly_log_path=anomaly_log_path,
                df_subset=df_subset,
                observed_residuals=observed_residuals_var,
                obs_mask_subset=obs_mask_subset,
                subset_vars=subset_vars,
                target=target,
                corr_threshold=cfg.corr_threshold,
                max_lag=cfg.max_lag,
                model="VAR",
            )

            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(residuals_var.index, residuals_var, marker="o", label="Residuals")
            ax.hlines([threshold, -threshold], xmin=residuals_var.index[0], xmax=residuals_var.index[-1],
                      linestyles="--", label="Threshold")
            if len(anomaly_indices) > 0:
                ax.scatter(anomaly_indices, observed_residuals_var.loc[anomaly_indices],
                           s=50, edgecolor="black", linewidth=0.5, label="Anomaly", zorder=5)
            ax.set_title(f"Residuals and Anomalies for '{target}' (VAR)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            set_x_ticks(ax, residuals_var.index, cfg.forecast_freq)
            _savefig(figures_dir / f"{target}_residuals_anomalies_subset.png")

        summary_results.append({
            "Compound": target,
            "Model": model_type,
            "Correlated_Variables": ", ".join(subset_vars),
            "Correlation_Details": " | ".join(correlation_info),
            "Lag_Order": lag_order_used if lag_order_used is not None else "",
            "Selected_Lag_Order": selected_lag_order,
            "Anomaly_Count": anomaly_count,
            "Notes": notes,
        })

    df_summary = pd.DataFrame(summary_results)
    summary_csv = output_dir / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    return {
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir),
        "summary_csv": str(summary_csv),
        "anomaly_log": str(anomaly_log_path),
    }
