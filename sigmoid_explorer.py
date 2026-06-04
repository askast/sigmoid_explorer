"""
Interactive asymmetric-sigmoid explorer.

Reads a CSV of (time, temperature), fits the same asymmetric sigmoid used in
`tank_sigmoid_fit.py` but directly in TIME coordinates (no tank size or flow
rate needed), and opens a matplotlib window with text entries / sliders for
each coefficient so the user can perturb the fit by hand and watch the curve
update live. The fit curve (at current parameter values) can be exported to
a CSV. A different input CSV can be loaded in-window without restarting.

If the input has more than two columns, a Tk dialog asks which column is time
and which is temperature; with two columns it just uses them in order.

Model (time-domain form):
    T(t) = T_low + (T_high - T_low) / [1 + exp(-k*(t - t_m))]^(1/nu)

Controls:
    T_low, T_high   numerical text entries (unbounded)
    k, t_m, nu      sliders
    Import CSV      pick a new input file
    Export fit      save the current fit curve as a CSV
    Reset to fit    restore the scipy fit

Usage:
    uv run python sigmoid_explorer.py                # opens file picker
    uv run python sigmoid_explorer.py path/to/file.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

# pyi_splash is injected by PyInstaller --splash at build time; it's not
# importable in a normal `python sigmoid_explorer.py` run.
try:
    import pyi_splash  # type: ignore
except ImportError:
    pyi_splash = None


def _close_splash():
    """Dismiss the PyInstaller splash (no-op when not running frozen)."""
    if pyi_splash is None:
        return
    try:
        if pyi_splash.is_alive():
            pyi_splash.close()
    except Exception:
        pass


def asymmetric_sigmoid(t, T_low, T_high, k, t_m, nu):
    """Same shape as tank_sigmoid_fit.asymmetric_sigmoid but in time coords."""
    z = -k * (np.asarray(t, dtype=float) - t_m)
    z = np.clip(z, -500.0, 500.0)
    factor = np.exp(-np.log1p(np.exp(z)) / nu)
    return T_low + (T_high - T_low) * factor


def load_csv_raw(path):
    """Read a CSV/TSV file with auto-detected delimiter and header.

    Returns (column_names, data_array). data_array is float; cells that
    don't parse as numbers become NaN. Column names come from the header
    row if one is detected; otherwise they're `col_0`, `col_1`, ...
    """
    # sep=None + python engine sniffs comma vs tab vs semicolon. header="infer"
    # would always treat row 0 as a header; instead we detect explicitly so
    # a fully numeric file isn't forced to lose its first row. utf-8-sig
    # strips the BOM Excel often writes; cp1252 is the Windows fallback for
    # files that store characters like ° as raw 0xb0 instead of UTF-8.
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None, dtype=str,
                         encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", header=None, dtype=str,
                         encoding="cp1252")
    first_row = df.iloc[0].tolist()
    try:
        [float(x) for x in first_row]
        column_names = [f"col_{i}" for i in range(df.shape[1])]
        body = df
    except (ValueError, TypeError):
        column_names = [(str(s).strip().lstrip("﻿") or f"col_{i}")
                        for i, s in enumerate(first_row)]
        body = df.iloc[1:].reset_index(drop=True)
    data = body.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return column_names, data


def _columns_to_t_T(data, t_idx, T_idx):
    """Pull two columns out of a raw data array, drop NaN rows, sort by time."""
    sub = data[:, [t_idx, T_idx]]
    mask = np.isfinite(sub).all(axis=1)
    sub = sub[mask]
    sub = sub[np.argsort(sub[:, 0])]
    return sub[:, 0], sub[:, 1]


def _guess_default_columns(names):
    """Pick sensible defaults for the time/temperature dropdowns.

    Time: prefer 'physical time' / 'elapsed' / time-with-explicit-seconds
    over a bare 'time' match; reject 'step'/'index'/'iteration'/'count'
    columns even if they contain the word 'time' (e.g. 'Time step').
    Temperature: prefer 'outlet' over generic 'temp' over 'inlet'; require
    a temperature marker; reject flow/rate columns."""
    lower = [n.lower() for n in names]

    def is_time(n):
        if any(tok in n for tok in ("step", "index", "iter", "count", "no.")):
            return False
        return "time" in n or n.strip() in ("t", "t (s)", "t(s)")

    t_candidates = [i for i, n in enumerate(lower) if is_time(n)]
    t_candidates.sort(key=lambda i: (
        0 if "physical" in lower[i] else
        1 if "elapsed" in lower[i] else
        2 if ("(s)" in lower[i] or "(sec" in lower[i] or "second" in lower[i]) else 3
    ))
    t_idx = t_candidates[0] if t_candidates else 0

    def is_temp(n):
        if any(tok in n for tok in ("flow", "rate", "gpm", "lpm")):
            return False
        return ("temp" in n or "outlet" in n or "inlet" in n
                or "°" in n or "( f" in n or "( c" in n)

    T_candidates = [i for i, n in enumerate(lower) if i != t_idx and is_temp(n)]
    T_candidates.sort(key=lambda i: (
        0 if "outlet" in lower[i] else
        1 if "temp" in lower[i] else
        2 if "inlet" in lower[i] else 3
    ))
    if T_candidates:
        return t_idx, T_candidates[0]
    # Fallback: any non-time column that isn't an obvious counter or flow.
    bad = ("step", "index", "iter", "count", "no.", "flow", "rate", "gpm", "lpm")
    for i, n in enumerate(lower):
        if i == t_idx:
            continue
        if any(tok in n for tok in bad):
            continue
        return t_idx, i
    # Last resort: first column that isn't the time column.
    for i in range(len(names)):
        if i != t_idx:
            return t_idx, i
    return t_idx, t_idx


def _pick_columns_dialog(column_names):
    """Modal Tk dialog with two dropdowns. Returns (t_idx, T_idx) or None."""
    import tkinter as tk
    from tkinter import ttk

    default_t, default_T = _guess_default_columns(column_names)
    options = [f"[{i}] {name}" for i, name in enumerate(column_names)]
    width = max(20, min(60, max(len(o) for o in options)))

    root = tk.Tk()
    root.title("Choose columns")
    root.resizable(False, False)
    result = {"selection": None}

    frame = tk.Frame(root, padx=14, pady=12)
    frame.pack()
    tk.Label(frame, text="Time column:").grid(row=0, column=0, sticky="e", pady=4, padx=(0, 8))
    tk.Label(frame, text="Temperature column:").grid(row=1, column=0, sticky="e", pady=4, padx=(0, 8))
    t_var = tk.StringVar(value=options[default_t])
    T_var = tk.StringVar(value=options[default_T])
    ttk.Combobox(frame, textvariable=t_var, values=options, width=width,
                 state="readonly").grid(row=0, column=1, pady=4)
    ttk.Combobox(frame, textvariable=T_var, values=options, width=width,
                 state="readonly").grid(row=1, column=1, pady=4)

    def on_ok(_=None):
        result["selection"] = (options.index(t_var.get()), options.index(T_var.get()))
        root.destroy()

    def on_cancel(_=None):
        root.destroy()

    btns = tk.Frame(root, pady=8)
    btns.pack()
    tk.Button(btns, text="OK", width=10, command=on_ok).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=6)

    root.bind("<Return>", on_ok)
    root.bind("<Escape>", on_cancel)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 3}")
    _raise_root(root)
    # Drop topmost once shown so it doesn't float over everything afterward.
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return result["selection"]


def load_csv_interactive(path):
    """Load a CSV and return (t, T). If the file has >2 columns, prompt the
    user via _pick_columns_dialog. Returns None if the dialog is cancelled."""
    names, data = load_csv_raw(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: need at least 2 columns (got {data.shape}).")
    if data.shape[1] == 2:
        t_idx, T_idx = 0, 1
    else:
        picked = _pick_columns_dialog(names)
        if picked is None:
            return None
        t_idx, T_idx = picked
    return _columns_to_t_T(data, t_idx, T_idx)


def initial_guess(t, T):
    T_low_0 = float(np.min(T))
    T_high_0 = float(np.max(T))
    span = T_high_0 - T_low_0 if T_high_0 > T_low_0 else 1.0
    midpoint = T_low_0 + 0.5 * span
    idx = int(np.argmin(np.abs(T - midpoint)))
    t_m_0 = float(t[idx])
    lo_idx = int(np.argmin(np.abs(T - (T_low_0 + 0.1 * span))))
    hi_idx = int(np.argmin(np.abs(T - (T_low_0 + 0.9 * span))))
    width = abs(t[hi_idx] - t[lo_idx])
    k_0 = 4.0 / width if width > 0 else 1.0
    return [T_low_0, T_high_0, k_0, t_m_0, 1.0]


def fit(t, T):
    p0 = initial_guess(t, T)
    t_span = max(t.max() - t.min(), 1e-6)
    T_min, T_max = float(np.min(T)), float(np.max(T))
    span = max(T_max - T_min, 1.0)
    lower = [T_min - span, T_min,        1e-6,            t.min() - t_span, 0.05]
    upper = [T_max,        T_max + span, 1e4 / t_span,    t.max() + t_span, 50.0]
    p0 = [min(max(p, lo + 1e-9), up - 1e-9) for p, lo, up in zip(p0, lower, upper)]
    popt, _ = curve_fit(asymmetric_sigmoid, t, T, p0=p0,
                        bounds=(lower, upper), maxfev=20000)
    return popt


def goodness(t, T, params):
    pred = asymmetric_sigmoid(t, *params)
    resid = T - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(T)))
    return r2, rmse


PARAM_NAMES = ["T_low", "T_high", "k", "t_m", "nu"]


def slider_ranges(t, popt):
    """Slider ranges for the three coefficients exposed as sliders."""
    t_span = max(t.max() - t.min(), 1.0)
    return {
        "k":   (max(1e-4, popt[2] * 0.05), popt[2] * 10.0),
        "t_m": (t.min() - 0.2 * t_span, t.max() + 0.2 * t_span),
        "nu":  (0.05, 10.0),
    }


def _raise_root(root):
    """Force a Tk window to the foreground on Windows.

    Native common dialogs (file open/save) inherit the topmost/focus state of
    their parent. Without this, the parent is hidden behind the active window
    and the dialog opens behind it too, with no taskbar button to find it by.
    """
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()


def _pick_open_csv():
    """Tk file dialog for an input CSV. Returns the chosen path or None."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    _raise_root(root)
    try:
        path = filedialog.askopenfilename(
            title="Select CSV (time, temperature)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def _pick_save_csv(default_name="fit_curve.csv"):
    """Tk save dialog for the exported fit curve. Returns path or None."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    _raise_root(root)
    try:
        path = filedialog.asksaveasfilename(
            title="Save fit curve as CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def export_fit_curve(path, t_min, t_max, params, n_points=600):
    t_grid = np.linspace(t_min, t_max, n_points)
    T_fit = asymmetric_sigmoid(t_grid, *params)
    header = (f"time,temperature_fit\n"
              f"# T_low={params[0]:.8g}, T_high={params[1]:.8g}, "
              f"k={params[2]:.8g}, t_m={params[3]:.8g}, nu={params[4]:.8g}")
    np.savetxt(path, np.column_stack([t_grid, T_fit]),
               delimiter=",", header=header, comments="")


def _show_window(csv_path, t, T):
    """Build and show the figure for one CSV. Returns (next_path, t, T) if
    the user clicked Import and loaded another file; None when the window
    closes normally."""
    if len(t) < 6:
        print(f"'{csv_path}': need at least ~6 data points to fit 5 parameters.",
              file=sys.stderr)
        return None

    popt = fit(t, T)
    r2, rmse = goodness(t, T, popt)
    print(f"Initial fit ({csv_path}):")
    for n, v in zip(PARAM_NAMES, popt):
        print(f"  {n:6s} = {v: .6g}")
    print(f"  R^2  = {r2:.5f}    RMSE = {rmse:.5g}")

    t_dense = np.linspace(t.min(), t.max(), 600)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    plt.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.45)

    ax.scatter(t, T, s=18, color="#444", alpha=0.7, label="Data", zorder=2)
    (line,) = ax.plot(t_dense, asymmetric_sigmoid(t_dense, *popt),
                      color="#0066cc", lw=2, label="Sigmoid", zorder=5)
    vline = ax.axvline(popt[3], color="#999", lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Asymmetric sigmoid — {Path(csv_path).name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    readout = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                      family="monospace", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", alpha=0.85))

    tlow_ax  = fig.add_axes([0.16, 0.35, 0.18, 0.04])
    thigh_ax = fig.add_axes([0.55, 0.35, 0.18, 0.04])
    tlow_box  = TextBox(tlow_ax,  "T_low ",  initial=f"{popt[0]:.6g}")
    thigh_box = TextBox(thigh_ax, "T_high ", initial=f"{popt[1]:.6g}")

    ranges = slider_ranges(t, popt)
    slider_specs = [("k", popt[2]), ("t_m", popt[3]), ("nu", popt[4])]
    slider_axes = [fig.add_axes([0.16, 0.28 - i * 0.05, 0.68, 0.03])
                   for i in range(3)]
    sliders = {
        name: Slider(ax_s, name, ranges[name][0], ranges[name][1], valinit=val)
        for ax_s, (name, val) in zip(slider_axes, slider_specs)
    }

    import_ax = fig.add_axes([0.16, 0.05, 0.14, 0.04])
    export_ax = fig.add_axes([0.34, 0.05, 0.14, 0.04])
    reset_ax  = fig.add_axes([0.86, 0.05, 0.10, 0.04])
    import_btn = Button(import_ax, "Import CSV")
    export_btn = Button(export_ax, "Export fit")
    reset_btn  = Button(reset_ax,  "Reset to fit")

    status_text = fig.text(0.50, 0.015, f"Loaded: {Path(csv_path).name}",
                           ha="center", va="bottom", fontsize=9, color="#555")

    current = list(popt)
    state = {"next": None}  # set to (path, t, T) when Import succeeds

    def update_readout(params):
        r2_v, rmse_v = goodness(t, T, params)
        lines = [f"{n} = {v: .6g}" for n, v in zip(PARAM_NAMES, params)]
        lines.append(f"R^2  = {r2_v:.5f}")
        lines.append(f"RMSE = {rmse_v:.5g}")
        readout.set_text("\n".join(lines))

    def redraw():
        line.set_ydata(asymmetric_sigmoid(t_dense, *current))
        vline.set_xdata([current[3], current[3]])
        update_readout(current)
        fig.canvas.draw_idle()

    def on_textbox(idx):
        def handler(text):
            try:
                current[idx] = float(text)
            except ValueError:
                return
            redraw()
        return handler

    def on_slider(idx):
        def handler(val):
            current[idx] = float(val)
            redraw()
        return handler

    tlow_box.on_submit(on_textbox(0))
    thigh_box.on_submit(on_textbox(1))
    sliders["k"].on_changed(on_slider(2))
    sliders["t_m"].on_changed(on_slider(3))
    sliders["nu"].on_changed(on_slider(4))

    def on_reset(_):
        tlow_box.set_val(f"{popt[0]:.6g}")
        thigh_box.set_val(f"{popt[1]:.6g}")
        sliders["k"].set_val(popt[2])
        sliders["t_m"].set_val(popt[3])
        sliders["nu"].set_val(popt[4])
        status_text.set_text(f"Loaded: {Path(csv_path).name}")
        fig.canvas.draw_idle()

    def on_import(_):
        new_path = _pick_open_csv()
        if not new_path:
            return
        # Validate load+fit before tearing down the window — keeps the current
        # session alive if the new file is unparseable or too short. The
        # column picker (if it fires) runs once here, not again in _show_window.
        try:
            loaded = load_csv_interactive(new_path)
        except Exception as exc:
            print(f"Failed to load '{new_path}': {exc}", file=sys.stderr)
            status_text.set_text(f"Failed to load: {Path(new_path).name}")
            fig.canvas.draw_idle()
            return
        if loaded is None:
            return  # user cancelled the column dialog
        t_new, T_new = loaded
        if len(t_new) < 6:
            status_text.set_text(f"Skipped (too few points): {Path(new_path).name}")
            fig.canvas.draw_idle()
            return
        try:
            fit(t_new, T_new)
        except Exception as exc:
            print(f"Fit failed for '{new_path}': {exc}", file=sys.stderr)
            status_text.set_text(f"Fit failed: {Path(new_path).name}")
            fig.canvas.draw_idle()
            return
        state["next"] = (new_path, t_new, T_new)
        plt.close(fig)

    def on_export(_):
        save_path = _pick_save_csv(default_name=f"{Path(csv_path).stem}_fit.csv")
        if not save_path:
            return
        try:
            export_fit_curve(save_path, float(t.min()), float(t.max()), current)
        except Exception as exc:
            print(f"Failed to write '{save_path}': {exc}", file=sys.stderr)
            status_text.set_text(f"Failed to save: {Path(save_path).name}")
            fig.canvas.draw_idle()
            return
        print(f"Wrote fit curve to {save_path}")
        status_text.set_text(f"Saved: {Path(save_path).name}")
        fig.canvas.draw_idle()

    reset_btn.on_clicked(on_reset)
    import_btn.on_clicked(on_import)
    export_btn.on_clicked(on_export)

    update_readout(popt)
    _close_splash()
    plt.show()
    return state["next"]


def run(csv_path):
    if csv_path is None:
        csv_path = _pick_open_csv()
        if not csv_path:
            print("No CSV selected.", file=sys.stderr)
            return
    try:
        loaded = load_csv_interactive(csv_path)
    except Exception as exc:
        print(f"Failed to load '{csv_path}': {exc}", file=sys.stderr)
        return
    if loaded is None:
        print("Column selection cancelled.", file=sys.stderr)
        return
    t, T = loaded
    while True:
        result = _show_window(csv_path, t, T)
        if result is None:
            return
        csv_path, t, T = result


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Interactive asymmetric-sigmoid fitter for time/temperature CSVs.")
    p.add_argument("csv", nargs="?", default=None,
                   help="CSV file with columns: time, temperature. "
                        "If omitted, a file picker dialog opens at startup.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args.csv)


if __name__ == "__main__":
    main()
