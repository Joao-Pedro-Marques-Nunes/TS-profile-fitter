# TS_profile_fitter

A structured pipeline for fitting AUG Thomson-scattering profiles across a shot
database. The project supports modified-tanh and monotonic spline fits, optional
physics/separatrix calculations, and separate EQH or IDE equilibrium analyses.

## Purpose

This project is designed to load AUG VTA Thomson-scattering data, preprocess
the measurements, fit both modified-tanh and joint monotonic spline profiles,
and optionally calculate physics-based separatrix estimates.

## Structure

- `src/ts_profile_fitter/`: core package modules
- `notebooks/`: exploratory notebooks for visualization and debugging
- `configs/`: configuration and parameter templates
- `scripts/`: execution scripts for running the database pipeline
- `tests/`: pytest suite (`pytest.ini` sets the test directory and import paths)
- `data/`: raw or staged input files
- `results/`: output tables, per-interval results, caches, and plots

Importing the package requires `aug_sfutils`, `get_functions`, and `matplotlib`,
so it needs the AUG analysis environment. `ipfnpytools` and `pysepest` are
optional; without them, the features that use them are unavailable. The
database pipeline also needs the VTA shotfiles and the `aug_separatrix/`
connection-length provider, which must sit next to this project folder
(`../aug_separatrix/`). The package can be installed in editable mode with
`pip install -e .`, which does not install the AUG-specific modules.

## Configuration files

| File | Output directory | Role |
| --- | --- | --- |
| `configs/pipeline_config.yaml` | `results/EQH/` | Template for new runs |
| `configs/pipeline_config_run_06_for_thesis.yaml` | `results/EQH_run_06_for_thesis/` | Thesis run 06 |

Paths in these files are relative to `configs/`. Run-specific files matching
`configs/pipeline_config_run_*.yaml` are git-ignored, except the thesis
configuration.

## Usage

1. Edit `configs/pipeline_config.yaml` and, when selecting a subset of shots,
   `configs/shot_list.csv`. The configured database pickle supplies the shot
   intervals and existing database signals. Set `equilibrium` to `EQH` or
   `IDE`; this value controls mapping, physics inputs, separatrix calculations,
   and connection-length data.
2. Run the pipeline:

```bash
python scripts/run_full_database.py --config configs/pipeline_config.yaml
```

To start a new run without manually changing every output path, generate a
run-specific configuration from the template:

```bash
python scripts/initialize_run.py \
	--config configs/pipeline_config.yaml \
	--output-dir ../results/EQH_run_02 \
	--config-output configs/pipeline_config_run_02.yaml
python scripts/run_full_database.py --config configs/pipeline_config_run_02.yaml
```

The initializer derives the fitted pickle and CSV, cache, summary-plot, and
failure-diagnostic paths from `--output-dir`, resolved relative to the template's
folder. Without `--in-place` it writes a new file (`--config-output`, default
`<config>_run.yaml`) with absolute paths. `--in-place` rewrites the template with
paths relative to it, so use it only when that rewrite is intended.

Each newly computed physics input and connection length is also appended to
`db_Lgil2024_physics_input.csv` and `db_Lgil2024_connection length.csv` in the
run output directory, whether or not the corresponding disk cache is enabled.
These are human-readable reference tables; the hashed cache files remain the
execution cache and are reused on later runs when caching is enabled.

Paths in the YAML file are resolved relative to the YAML file, so the example
configuration can be launched from any working directory.

The canonical interval schema is `shot_id`, `t_begin`, `t_end`, and `label`;
times are in seconds. The loader also accepts the legacy raw-database names
`shot`, `t_begin (s)`, `t_end (s)`, and `profile_type` at the file-ingestion
boundary. The legacy database can therefore be used directly without creating
an intermediate `intervals.csv`.

The reusable entry points are:

```python
from ts_profile_fitter.pipeline import run_interval, run_shot, run_database
```

`run_interval()` returns the fit arrays and parameters for one interval.
`run_shot()` writes one JSON result per interval and an interval status CSV.
Failures are recorded per interval so one bad interval does not discard the
other results for that shot. `run_database()` returns the input database with
scalar fit and separatrix summaries merged by `(shot_id, t_begin, t_end)`.
The command-line entry point writes this enriched table as both pickle and CSV.
Detailed fitted profiles, covariance arrays, and physics grids remain in the
per-interval JSON files rather than being placed in CSV cells.
At the end of a full database run, `run_summary.json` is written in the output
directory. It contains the wall-clock runtime, total/successful/failed interval
counts, success rate, aggregate interval timing, and a compact per-shot summary.

The batch configuration can enable independent disk caches for shot-wide mapped
TS data (`cache.mapped_data`), physics inputs (`cache.physics_inputs`), and
connection-length calculations (`cache.connection_length`). The mapped-data cache
performs the expensive VTA-to-`rho_pol` and midplane mapping once per shot;
interval selection, channel trimming, outlier filtering, and fit weighting
remain interval-specific. Cache files are written under `cache.directory`
(default `<output_dir>/cache`) and are written atomically. Keys carry a version
tag (`mapped_data_v1`, `physics_inputs_v2`, `connection_length_v1`), so entries
under an older tag are not reused: the thesis configuration recomputes its
physics inputs on the first run after the `physics_inputs_v2` change, which needs
AUG data access. Failed computations are not reused unless `reuse_failed: true`.
Only plain serialized data is cached; live shotfile objects are never stored, and
the source database is not modified.

Outputs and caches are isolated by `output_dir`: the template writes to
`results/EQH/`, and the thesis configuration writes to
`results/EQH_run_06_for_thesis/`. Use a separate `output_dir` for IDE runs. The
connection-length cache key includes the equilibrium and provider ID, so EQH and
IDE values are not mixed. The thesis configuration sets
`cache.provider_id: get_LcSOL_selectable_v1`, the provider that
`scripts/run_full_database.py` passes to `run_database`. The template still sets
`get_LcSOL_v1`.

During execution the pipeline prints the current shot, interval number, time
window, elapsed time, and separatrix result. Interval failures are printed and
also retained in each shot's `interval_status.csv`.

A run writes the following layout under its `output_dir`, with one folder per
shot. Cache and plot folders appear only when those features are enabled:

```text
<output_dir>/
	shot_35970/
		interval_2.600_2.698.json
		interval_status.csv
	cache/
	profile_summary/
	profile_diagnostics/
	run_summary.json
	34383---38808_database_dt-0.1s_fitted.pkl
	34383---38808_database_dt-0.1s_fitted.csv
```

Set `fit_config.plots.enabled: true` to save figures. Under `fit_config.plots`,
`summary_directory` receives one `shot_<id>/` folder per shot, and
`diagnostics_directory` receives failure diagnostics when
`failure_diagnostics: true`. Figure names follow
`shot{id}_{equilibrium}_k{knots:02d}_t{begin}-{end}_{plot_kind}`, with the
interval times written with `p` in place of the decimal point, for example
`shot32959_EQH_k09_t0p500-0p600_te_comparison_full.png`.

Physics-based separatrix calculations require the site-specific compiled
field-line routines. The runner imports `get_LcSOL` from
`../aug_separatrix/get_connection_length_selectable.py` (the `aug_separatrix`
folder next to this project), which accepts `equilibrium='EQH'` or
`equilibrium='IDE'`. Set `physics.enabled: true` and ensure the selected
equilibrium shotfiles are available. If the equilibrium shotfile is unavailable
(`EquilibriumUnavailableError`), all intervals of that shot are recorded as
`skipped` and the database run continues.

The cold-dense edge filter (`cold_dense_edge_config`) removes edge points with
`Te < te_max` and `Ne > ne_min` when it is enabled; the thesis values are
`te_max: 1.1` (eV) and `ne_min: 1e19` (m^-3). It is disabled in the template and
in the thesis configuration. Modified-tanh settings: the thesis configuration uses
`deg_tail: 1` and `deg_top: 3` for both Te and Ne, with `fix_a_tail: 0` for Te.
The template uses `deg_tail: 1`, `deg_top: 3` for Te and `deg_tail: 2`,
`deg_top: 3` for Ne.

Use `notebooks/single_shot_and_interval.ipynb` to inspect a single interval
and compare batch results with the reference plotting workflow.

## Plotting

Plots are drawn with Matplotlib. Two helpers come from the optional `ipfnpytools`
package: `ipfnpytools.plot.plots` (used by the plotting functions) and
`ipfnpytools.save_figure` (used when figures are saved). Without `ipfnpytools`,
those functions raise `ImportError` when called, and saving raises it when
figures are requested.

## Validation

Run the tests from the project root in the active conda environment:

```bash
python -m pytest -q
```

`pytest.ini` sets the test directory and the import paths (`.` and `src/`).
`tests/test_pipeline.py` also imports `scripts/scan_spline_knots.py`, which needs
`aug_separatrix/` next to this folder.

Current result: `9 failed, 48 passed`, with 3 failures in each of
`tests/test_fitting.py`, `tests/test_plotting.py`, and `tests/test_preprocess.py`.
Some are test/code mismatches (mtanh parameter count, plot label and annotation
text). The others need investigation: outlier counts, NaN and shape errors in the
spline fits, a plotting call with a short parameter vector, and a shape mismatch
in the external `ipfnpytools` mapping. Treat a clean run as the release gate only
after these are resolved.
