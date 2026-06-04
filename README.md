# Sigmoid Explorer

Interactive asymmetric-sigmoid fitter for `(x, y)` data supplied as a CSV.

It reads a CSV, fits an asymmetric sigmoid, and opens a matplotlib window with
text entries and sliders for each coefficient so you can perturb the fit by hand
and watch the curve update live. The current curve can be exported to a CSV, and
a different input file can be loaded in-window without restarting.

This tool was built with **temperature mixing in tanks** in mind — fitting a
temperature-vs-time profile (e.g. the outlet temperature of a thermal storage
tank as it charges or discharges) directly in time coordinates, with no tank
size or flow rate required. The axes are labelled "Time" and "Temperature" for
that reason. But nothing in the math is specific to temperature: it is a
**general-purpose asymmetric-sigmoid fit** and works for any two-column dataset
that follows an S-shaped transition between two levels.

## Model

```
y(x) = y_low + (y_high - y_low) / [1 + exp(-k*(x - x_m))]^(1/nu)
```

| Parameter | Meaning                                            | Control    |
| --------- | -------------------------------------------------- | ---------- |
| `T_low`   | lower asymptote                                    | text entry |
| `T_high`  | upper asymptote                                    | text entry |
| `k`       | growth rate / steepness                            | slider     |
| `t_m`     | midpoint location along the x-axis                 | slider     |
| `nu`      | asymmetry (`nu = 1` is the symmetric logistic)     | slider     |

An initial fit is computed with `scipy.optimize.curve_fit`; the sliders and text
boxes then let you adjust each coefficient. The readout shows the live R² and
RMSE against the data, and **Reset to fit** restores the computed fit.

## Install & run

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.11):

```bash
uv sync

uv run python sigmoid_explorer.py                 # opens a file picker at startup
uv run python sigmoid_explorer.py path/to/file.csv
```

### Input format

A CSV (comma, tab, or semicolon delimited — auto-detected) with at least two
columns. With exactly two columns they are used in order as `(x, y)`. With more
than two columns, a dialog lets you choose which column is which; it pre-selects
sensible defaults by inspecting the header names.

### Controls

- **Import CSV** — load a different input file without restarting.
- **Export fit** — save the current fit curve (densely sampled) as a CSV, with
  the parameter values written into the header.
- **Reset to fit** — discard manual tweaks and restore the computed fit.

## Building a standalone executable

A [PyInstaller](https://pyinstaller.org/) spec is included to build a windowed
Windows executable (with a splash screen) into `dist/`:

```bash
uv sync                                    # installs the dev group, incl. pyinstaller
uv run pyinstaller sigmoid_explorer.spec
```
