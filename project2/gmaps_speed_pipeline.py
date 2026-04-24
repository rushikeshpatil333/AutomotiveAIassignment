import re
import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import pytesseract
from scipy.signal import savgol_filter

# New: plotting
import matplotlib
matplotlib.use("Agg")  # headless backend (works in terminals/servers)
import matplotlib.pyplot as plt

# --- Windows quick fix: point pytesseract to the Tesseract executable ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ----------------------------------------------------------------------


@dataclass
class ROI:
    """ROI in normalized coordinates (0..1): x,y,w,h relative to frame width/height."""
    x: float
    y: float
    w: float
    h: float


@dataclass
class Config:
    # Sampling
    target_hz: float = 10.0

    # Speed validity constraints
    speed_min_kph: int = 0
    speed_max_kph: int = 120

    # Outlier rejection (implied acceleration)
    amax_ms2: float = 10.0

    # Gap filling
    max_interp_gap_s: float = 1.0

    # Smoothing
    sg_window: int = 11
    sg_poly: int = 2

    # OCR
    tesseract_psm: int = 7

    # ROI (bottom-left speed badge; adjust if needed)
    speed_roi: ROI = field(default_factory=lambda: ROI(x=0.02, y=0.76, w=0.20, h=0.20))

    # Debug/progress
    print_every_n_samples: int = 50        # prints every N samples
    max_seconds: Optional[float] = None    # None => full video; set e.g. 30.0 for testing


def _crop_norm(frame: np.ndarray, roi: ROI) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = int(roi.x * w)
    y1 = int(roi.y * h)
    x2 = int((roi.x + roi.w) * w)
    y2 = int((roi.y + roi.h) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2].copy()


def _preprocess_variants(crop_bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    variants = []
    for img in (norm, 255 - norm):
        thr = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 7
        )
        thr = cv2.medianBlur(thr, 3)
        variants.append(thr)
    return variants


def _ocr_digits(bin_img: np.ndarray, psm: int) -> tuple[Optional[int], str]:
    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(bin_img, config=config).strip()
    text = re.sub(r"\s+", "", text)

    m = re.search(r"(\d{1,3})", text)
    if not m:
        return None, text
    return int(m.group(1)), text


def read_speed_from_frame(frame_bgr: np.ndarray, cfg: Config) -> Dict:
    crop = _crop_norm(frame_bgr, cfg.speed_roi)
    variants = _preprocess_variants(crop)

    best_val = None
    best_raw = ""
    for v in variants:
        val, raw = _ocr_digits(v, cfg.tesseract_psm)
        if val is None:
            continue
        if not (cfg.speed_min_kph <= val <= cfg.speed_max_kph):
            continue
        best_val = val
        best_raw = raw
        break

    return {"speed_kph": best_val, "ocr_raw": best_raw}


def sample_video(video_path: str, cfg: Config) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / cfg.target_hz)))

    rows = []
    frame_idx = 0
    sample_idx = 0
    start_t = None

    print(f"Opened video: {video_path}")
    print(f"FPS={fps:.3f}  target_hz={cfg.target_hz}  step={step}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if start_t is None:
                start_t = t_sec

            # Stop early for testing
            if cfg.max_seconds is not None and (t_sec - start_t) > cfg.max_seconds:
                print(f"Stopping early at t={t_sec:.2f}s (max_seconds={cfg.max_seconds})")
                break

            res = read_speed_from_frame(frame, cfg)
            rows.append({
                "sample_idx": sample_idx,
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "speed_kph_raw": res["speed_kph"],
                "ocr_raw": res["ocr_raw"],
            })

            sample_idx += 1
            if cfg.print_every_n_samples and (sample_idx % cfg.print_every_n_samples == 0):
                print(f"Processed samples={sample_idx}  t={t_sec:.2f}s  last_speed={res['speed_kph']}")

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows).sort_values("t_sec").reset_index(drop=True)


def clean_and_resample(df_raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_raw.copy()
    df.loc[~df["speed_kph_raw"].between(cfg.speed_min_kph, cfg.speed_max_kph), "speed_kph_raw"] = np.nan

    if df.empty:
        raise RuntimeError("No samples were collected from the video.")

    t0, t1 = df["t_sec"].min(), df["t_sec"].max()
    dt = 1.0 / cfg.target_hz
    t_grid = np.arange(t0, t1 + 1e-9, dt)

    s = pd.Series(df["speed_kph_raw"].values, index=df["t_sec"].values)
    s = s[~s.index.duplicated(keep="first")]

    median = np.nanmedian(s.values) if np.isfinite(np.nanmedian(s.values)) else 0.0
    speed = np.interp(t_grid, s.index.values, np.nan_to_num(s.values, nan=median))
    out = pd.DataFrame({"t_sec": t_grid, "speed_kph": speed})

    # Drop spikes using implied acceleration
    v_ms = out["speed_kph"].values / 3.6
    a = np.diff(v_ms) / dt
    bad = np.zeros_like(v_ms, dtype=bool)
    bad[1:] |= np.abs(a) > cfg.amax_ms2
    out.loc[bad, "speed_kph"] = np.nan

    # Interpolate short gaps only
    max_gap_n = int(round(cfg.max_interp_gap_s * cfg.target_hz))
    out["speed_kph"] = out["speed_kph"].interpolate(limit=max_gap_n, limit_direction="both")

    # Smooth
    out["speed_kph_smooth"] = out["speed_kph"]
    if out["speed_kph_smooth"].notna().sum() >= cfg.sg_window:
        filled = out["speed_kph_smooth"].interpolate(limit_direction="both").values
        w = cfg.sg_window if cfg.sg_window % 2 == 1 else cfg.sg_window + 1
        out["speed_kph_smooth"] = savgol_filter(filled, window_length=w, polyorder=cfg.sg_poly, mode="interp")

    return out


def compute_dynamics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    dt = 1.0 / cfg.target_hz

    v_ms = out["speed_kph_smooth"].values / 3.6
    out["v_ms"] = v_ms

    dist = np.zeros_like(v_ms, dtype=float)
    dist[1:] = np.cumsum(0.5 * (v_ms[1:] + v_ms[:-1]) * dt)
    out["distance_m"] = dist

    a = np.full_like(v_ms, np.nan, dtype=float)
    a[1:-1] = (v_ms[2:] - v_ms[:-2]) / (2 * dt)
    out["a_ms2"] = a

    return out


def save_distance_time_png(df_dyn: pd.DataFrame, png_path: str, title: str = "Distance vs Time"):
    if df_dyn.empty:
        raise RuntimeError("Dynamics dataframe is empty; cannot plot.")

    needed = {"t_sec", "distance_m"}
    missing = needed - set(df_dyn.columns)
    if missing:
        raise ValueError(f"Cannot plot; missing columns: {sorted(missing)}")

    plot_df = df_dyn[["t_sec", "distance_m"]].dropna().sort_values("t_sec")
    if plot_df.empty:
        raise RuntimeError("No data to plot after dropping NaNs.")

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["t_sec"], plot_df["distance_m"], linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(png_path, dpi=200)
    plt.close()


def main(video_path: str, out_prefix: str = "output", make_png: bool = True):
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        raise RuntimeError(
            "Tesseract executable not found at:\n"
            f"  {pytesseract.pytesseract.tesseract_cmd}\n"
            "Update pytesseract.pytesseract.tesseract_cmd in this script."
        )

    cfg = Config()

    df_raw = sample_video(video_path, cfg)
    raw_csv = f"{out_prefix}_raw_samples.csv"
    df_raw.to_csv(raw_csv, index=False)

    df_clean = clean_and_resample(df_raw, cfg)
    clean_csv = f"{out_prefix}_speed_clean.csv"
    df_clean.to_csv(clean_csv, index=False)

    df_dyn = compute_dynamics(df_clean, cfg)
    dyn_csv = f"{out_prefix}_dynamics.csv"
    df_dyn.to_csv(dyn_csv, index=False)

    png_path = f"{out_prefix}_distance_vs_time.png"
    if make_png:
        save_distance_time_png(df_dyn, png_path, title=f"{out_prefix}: Distance vs Time")

    print("Done. Wrote:")
    print(f"  {raw_csv}")
    print(f"  {clean_csv}")
    print(f"  {dyn_csv}")
    if make_png:
        print(f"  {png_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-png", action="store_true", help="Do not generate distance-vs-time PNG")
    args = ap.parse_args()
    main(args.video, args.out, make_png=(not args.no_png))
    