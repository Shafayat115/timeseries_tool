import io
import re
import tempfile
from pathlib import Path
import zipfile

import pandas as pd
import streamlit as st

from ts_anomaly.pipeline import run_pipeline, PipelineConfig


def _safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:80] if s else "file"


def _zip_dir(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(folder))
    return buf.getvalue()


def _peek_csv_has_date(upload) -> tuple[bool, str]:
    try:
        upload.seek(0)
        df = pd.read_csv(upload, nrows=5)
        upload.seek(0)
    except Exception as e:
        return False, f"Could not read CSV: {e}"
    if "Date" not in df.columns:
        return False, "Missing required column: Date"
    return True, ""


@st.cache_data(show_spinner=False)
def _run_cached(files_bytes: list[tuple[bytes, str, str]], cfg: PipelineConfig) -> dict:
    """
    files_bytes: [(raw_bytes, original_filename, category), ...]
    Writes outputs to a temp dir and returns zipped outputs + summary/anomaly content.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "outputs"

        # recreate file-like objects from bytes for pandas
        file_specs = []
        for raw, fname, cat in files_bytes:
            bio = io.BytesIO(raw)
            bio.name = fname
            file_specs.append((bio, cat))

        paths = run_pipeline(file_specs, outdir, cfg)

        # bundle everything for Streamlit to download
        summary_bytes = Path(paths["summary_csv"]).read_bytes()
        anomaly_bytes = Path(paths["anomaly_log"]).read_bytes()
        zip_bytes = _zip_dir(Path(paths["output_dir"]))

        # collect pngs to display
        pngs = sorted(Path(paths["figures_dir"]).glob("*.png"))
        png_payloads = [(p.name, p.read_bytes()) for p in pngs]

        return {
            "summary_bytes": summary_bytes,
            "anomaly_bytes": anomaly_bytes,
            "zip_bytes": zip_bytes,
            "pngs": png_payloads,
        }


def main():
    st.set_page_config(page_title="TS Anomaly Tool", layout="wide")
    st.title("Time-series Forecast + Anomaly Detection")

    st.write("Upload one or more CSVs with a **Date** column and one or more numeric columns (example: Drugs).")

    uploads = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

    with st.sidebar:
        st.header("Parameters")
        forecast_steps = st.number_input("Forecast steps", min_value=1, max_value=365, value=12)
        forecast_freq = st.text_input("Forecast frequency", value="2D")
        max_lag = st.number_input("Max lag", min_value=1, max_value=30, value=2)
        corr_threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.6)
        interpolation_method = st.selectbox("Interpolation method", ["pchip", "polynomial", "linear"], index=0)
        interpolation_order = st.number_input("Polynomial order", min_value=1, max_value=5, value=2)

    # validate uploads
    if uploads:
        bad = []
        for up in uploads:
            ok, msg = _peek_csv_has_date(up)
            if not ok:
                bad.append((up.name, msg))
        if bad:
            st.error("Some files are not valid. Fix these and re-upload:")
            for name, msg in bad:
                st.write(f"- {name}: {msg}")
            return

    run_btn = st.button("Run analysis", disabled=not uploads)

    if uploads and run_btn:
        st.subheader("Category names (prefixes)")
        cats = []
        for up in uploads:
            default_cat = _safe_name(Path(up.name).stem)
            cat = st.text_input(f"Category for {up.name}", value=default_cat, key=f"cat_{up.name}")
            cats.append(_safe_name(cat))

        cfg = PipelineConfig(
            forecast_steps=int(forecast_steps),
            forecast_freq=str(forecast_freq),
            max_lag=int(max_lag),
            corr_threshold=float(corr_threshold),
            interpolation_method=str(interpolation_method),
            interpolation_order=int(interpolation_order),
        )

        # build stable cache key payload
        files_bytes = []
        for up, cat in zip(uploads, cats):
            up.seek(0)
            raw = up.read()
            files_bytes.append((raw, up.name, cat))

        with st.spinner("Running pipeline..."):
            payload = _run_cached(files_bytes, cfg)

        st.success("Done")

        st.subheader("summary.csv")
        summary_df = pd.read_csv(io.BytesIO(payload["summary_bytes"]))
        st.dataframe(summary_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Download summary.csv", payload["summary_bytes"], "summary.csv", "text/csv")
        with col2:
            st.download_button("Download anomalies.txt", payload["anomaly_bytes"], "anomalies.txt", "text/plain")
        with col3:
            st.download_button("Download outputs.zip", payload["zip_bytes"], "outputs.zip", "application/zip")

        st.subheader("Plots")
        pngs = payload["pngs"]
        if not pngs:
            st.info("No PNGs were produced.")
        else:
            cols = st.columns(3)
            for i, (name, b) in enumerate(pngs):
                with cols[i % 3]:
                    st.image(b, caption=name, use_container_width=True)
