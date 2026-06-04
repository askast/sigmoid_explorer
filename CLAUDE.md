# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file desktop tool (`sigmoid_explorer.py`, ~500 lines) that fits an asymmetric sigmoid to a `(time, temperature)` CSV and opens a live matplotlib window where the user hand-tunes the five fit parameters via text boxes / sliders, then exports the resulting curve. Extracted from a larger `tank_sigmoid_fit.py` project — the model fits directly in time coordinates (no tank size or flow rate).

## Commands

```bash
uv run python sigmoid_explorer.py                 # opens a Tk file picker at startup
uv run python sigmoid_explorer.py path/to/file.csv

uv sync                                            # install deps (incl. dev group)
uv run pyinstaller sigmoid_explorer.spec          # build the windowed Windows .exe -> dist/
```

There is no test suite, linter config, or CI. Dependencies are managed by `uv` (`pyproject.toml` + `uv.lock`); Python >= 3.11.

## Model

```
T(t) = T_low + (T_high - T_low) / [1 + exp(-k*(t - t_m))]^(1/nu)
```

Five params, in the fixed order used throughout (`PARAM_NAMES`): `T_low, T_high, k, t_m, nu`. `T_low`/`T_high` are exposed as unbounded text boxes; `k`/`t_m`/`nu` as sliders. The `asymmetric_sigmoid` implementation clips the exponent and uses `log1p`/`exp` for numerical stability — keep that form if you touch it.

## Architecture / flow

The control flow is a loop in `run()`: load → `_show_window()` → if the user clicks **Import CSV**, the window returns the next `(path, t, T)` and the loop reopens; closing the window normally returns `None` and exits. This is how a new file is loaded without restarting the process.

Pipeline within one session:
- `load_csv_raw` — delimiter/header auto-detection (comma/tab/semicolon via `sep=None`), explicit numeric-first-row detection so fully numeric files don't lose row 0, and `utf-8-sig` → `cp1252` encoding fallback (Excel BOM / raw `°`).
- `load_csv_interactive` — for >2 columns, raises a Tk dropdown dialog (`_pick_columns_dialog`); 2-column files skip it. Returns `None` when the user cancels — callers must treat `None` as "cancelled", distinct from an exception.
- `_guess_default_columns` — heuristic defaults for the dropdowns (keyword scoring; rejects `step`/`index`/`flow`/`rate` columns). Pure string matching on header names.
- `initial_guess` → `fit` (scipy `curve_fit` with computed bounds) → `goodness` (R²/RMSE).
- `_show_window` builds the matplotlib figure and wires widget callbacks that mutate a shared `current` list and call `redraw()`.

### Things that bite

- **Tk + matplotlib coexist.** File pickers and the column dialog are raw Tkinter (`tk.Tk()` created and destroyed per dialog) while the main window is matplotlib widgets. Don't assume one GUI toolkit.
- **Tk dialogs must be forced to the foreground.** On Windows a withdrawn/un-raised root spawns its dialog *behind* the active window with no taskbar button to recover it. Every dialog calls `_raise_root()` (sets `-topmost` + `lift()` + `focus_force()`) before opening; the visible column picker also drops `-topmost` via `after()` once shown so it doesn't permanently float. Reuse `_raise_root()` for any new Tk dialog.
- **`_show_window` validates a new file's load *and* fit before tearing down the current window** (in `on_import`), so a bad import leaves the existing session intact. Preserve that "validate before close" ordering if you refactor import.
- **PyInstaller splash.** `pyi_splash` is injected only in the frozen build; the `try/except ImportError` guard and `_close_splash()` are no-ops in a plain `python` run. `splash.png` is the splash image referenced by `sigmoid_explorer.spec`.
- The `.spec` file is the source of truth for the build (windowed, UPX, splash) — edit it, not raw `pyinstaller` flags.
