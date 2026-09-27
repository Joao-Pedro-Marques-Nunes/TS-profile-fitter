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
- `data/`: raw or staged input files
- `results/`: output tables, per-interval results, caches, and plots

The pipeline is intended to run in an AUG analysis environment with access to
`aug_sfutils`, the VTA shotfiles, and the compiled field-line routines used by
the connection-length provider. The package-level fitting and preprocessing
tests can still be run without those site-specific dependencies.

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
failure-diagnostic paths from `output_dir`. By default it leaves the template
unchanged; use `--in-place` only when rewriting the supplied YAML is intended.

When physics or connection-length caching is enabled, newly computed cache
entries are also appended to `db_Lgil2024_physics input.csv` and
`db_Lgil2024_connection length.csv` in the run output directory. These are
human-readable reference tables; the hashed cache files remain the execution
cache and are reused on later runs.

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
TS data, physics inputs, and connection-length calculations. The mapped-data cache
performs the expensive VTA-to-`rho_pol` and midplane mapping once per shot;
interval selection, channel trimming, outlier filtering, and fit weighting
remain interval-specific. Cache files are written under `results/cache/`, use
configuration-aware keys, and are written atomically. Failed computations are
not reused by default. Only plain serialized data is cached; live shotfile
objects are never stored, and the source database is not modified.

For the supplied configuration, outputs and caches are isolated under
`results/EQH/` or `results/IDE/`. The connection-length cache key includes the
selected equilibrium and provider ID, so EQH and IDE results are not mixed.
Use `cache.provider_id: get_LcSOL_selectable_v1` with the selectable provider.

During execution the pipeline prints the current shot, interval number, time
window, elapsed time, and separatrix result. Interval failures are printed and
also retained in each shot's `interval_status.csv`.

The default output layout is:

```text
results/
	shot_32959/
		interval_0.500_0.600.json
		interval_status.csv
	run_summary.json
	34383---38808_database_dt-0.1s_fitted.pkl
	34383---38808_database_dt-0.1s_fitted.csv
```

Set `fit_config.plots.enabled: true` to save comparison figures alongside the
JSON results. Figure names include the shot, equilibrium, knot count, and
actual interval bounds, for example:
`shot32959_EQH_k09_t0p500-0p600_te_comparison_full.png`.

Physics-based separatrix calculations require the site-specific compiled
field-line routines. The supplied runner uses
`aug_separatrix/get_connection_length_selectable.py`, whose `get_LcSOL`
function accepts `equilibrium='EQH'` or `equilibrium='IDE'`. Set
`physics.enabled: true` and ensure the selected equilibrium shotfiles are
available. If an IDE shotfile is unavailable, that shot is recorded as
`skipped` rather than aborting the complete database run.

The active configuration also applies a cold-dense edge filter before fitting:
edge points are retained when `Te <= 1.1 eV` and `Ne >= 1e19 m^-3`. The current
global modified-tanh settings use linear tails and quadratic top polynomials
(`deg_tail: 1`, `deg_top: 2`) for both Te and Ne.

Use `notebooks/single_shot_and_interval.ipynb` to inspect a single interval
and compare batch results with the reference plotting workflow.

## Plotting

This package uses `ipfnpytools.plot.plots` for visualization, because it is a wrapper around Matplotlib with your existing plot conventions.

## Validation

Run the local tests with:

```bash
source /shares/users/work/joanune/Desktop/.venv/bin/activate
pytest -q
```

The current suite has two known failures: the mtanh test still asserts the
obsolete six-parameter return shape, and the synthetic `ipfnpytools` mapping
fixture does not match the external package's expected equilibrium array
dimensions. These should be updated before treating a clean test run as a
release gate.
