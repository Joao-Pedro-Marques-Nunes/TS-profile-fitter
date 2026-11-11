# TS_profile_fitter

Fits AUG Thomson-scattering (TS) electron density (Ne) and temperature (Te) profiles for every time interval in a database. Each interval gets a joint monotonic spline fit and an independent modified-tanh (mtanh) fit per quantity. Optional physics steps estimate separatrix and pedestal quantities. Each run uses one equilibrium, EQH or IDE.

## Purpose

This project is designed to load AUG VTA Thomson-scattering data, preprocess
the measurements, fit both modified-tanh and joint monotonic spline profiles,
and optionally calculate physics-based separatrix estimates.

## Requirements

Python packages, installed with `pip install -e .` (see `setup.py`): numpy, pandas, scipy, statsmodels, pyyaml, matplotlib.

These are not installed by `setup.py`:

| Requirement | Used by | When it is needed |
| --- | --- | --- |
| `aug_sfutils`, `get_functions` | Shotfile access in `loading.py` and `preprocess.py`. `loading.py` adds `/shares/departments/AUG/users/dsil/pymodules` to `sys.path` | Import time |
| `ipfnpytools` | rho_pol mapping (`mapping_method: ipfnpytools`), `ipfn_spline`, the equilibrium separatrix, figures | Call time. The imports are guarded, so the package imports without it |
| `pysepest` | Separatrix estimate (`separatrix.py`) | Call time, when `physics.enabled` is true |
| `aug_separatrix/`, module `get_connection_length_selectable` | Connection-length provider `get_LcSOL`, imported by `scripts/run_full_database.py` and `scripts/scan_spline_knots.py` | Import time of those scripts. The folder must sit next to the repository (`../aug_separatrix/`) |

The thesis configuration needs all four. `ipfnpytools` and `pysepest` are also local folders in the workspace (`../ipfnpytools/`, `../pysepest/`); they must be importable in the active environment. The pipeline also needs access to the AUG shotfiles (VTA, the equilibrium and the power signals).

## Repository layout

```
configs/                 YAML templates and run configurations; shot_list.csv
scripts/                 run_full_database.py and helper scripts (see Auxiliary scripts)
src/ts_profile_fitter/   package modules (listed below)
tests/                   pytest suite (57 tests)
data/                    input interval database (git-ignored; not in the repository)
results/                 run outputs; only the thesis fitted pickle is tracked
notebooks/               exploratory notebooks
```

Package modules in `src/ts_profile_fitter/`:

- `pipeline.py`: database, shot and interval orchestration; JSON, status and summary output
- `preprocess.py`: mapping of VTA data to rho_pol and midplane R; validity, window and statistical outlier filters
- `fitting.py`: knot placement, `ondrej_spline` and `ipfn_spline` fits, modified-tanh fits
- `loading.py`: shotfile access, shot-list and interval loaders, physics inputs
- `derived_quantities.py`: equilibrium separatrix at a given rho_pol
- `separatrix.py`: pysepest wrapper
- `connection_length.py`: connection-length provider wrapper
- `fit_metrics.py`: SSE, weighted SSE and reduced chi-square
- `cache.py`: versioned disk caches
- `plotting.py`, `figure_naming.py`: figures and file names

## Configuration files

| File | Output directory | Role |
| --- | --- | --- |
| `configs/pipeline_config.yaml` | `results/EQH/` | Template for new runs |
| `configs/pipeline_config_run_06_for_thesis.yaml` | `results/EQH_run_06_for_thesis/` | Thesis run 06; the reference for the values listed below |

Paths in these files are relative to `configs/`. Both files use the same keys, but their values differ, so compare them before starting a new run. Run-specific files matching `configs/pipeline_config_run_*.yaml` are git-ignored, except the thesis configuration.

## Quick start

Run the thesis configuration, which writes to `results/EQH_run_06_for_thesis/`:

```bash
python scripts/run_full_database.py --config configs/pipeline_config_run_06_for_thesis.yaml
```

Start a new run from the template:

```bash
python scripts/initialize_run.py --config configs/pipeline_config.yaml --output-dir ../results/EQH_run_02 --config-output configs/pipeline_config_run_02.yaml
python scripts/run_full_database.py --config configs/pipeline_config_run_02.yaml
```

`initialize_run.py` options:

- `--config`: template YAML (required).
- `--output-dir`: root of all derived outputs (required). Relative paths are resolved against the template folder.
- `--config-output`: destination of the generated file (default `<config>_run.yaml` next to the template).
- `--in-place`: rewrite the template instead of writing a copy.

Without `--in-place`, the input paths `data_dir`, `shot_list`, `database` and `intervals` are written as absolute paths. The output paths are derived from `--output-dir`: `output_dir`, `output_database` and `output_database_csv` (named `<database stem>_fitted.pkl` and `.csv`), `cache.directory`, and the summary and diagnostics plot folders. The plot folders are written under `fit_config.plots`, so keep plot settings there; a top-level `plots` section would override them. With `--in-place`, input paths are written relative to the template folder.

## Pipeline

One run executes the steps below. Function names refer to `src/ts_profile_fitter/pipeline.py` unless stated otherwise.

1. **Entry point** (`scripts/run_full_database.py`). Reads the YAML, builds the `fit_config` mapping (see the YAML section below), loads the shot list and the interval table, and calls `run_database` with `get_LcSOL` as the connection-length provider. The enriched table is saved as `output_database` (pickle) and `output_database_csv` (CSV).
2. **Database** (`run_database`). Creates the disk caches, runs `run_shot` for each shot in the shot list, and returns the enriched table. `run_summary.json` is written at the end of the run.
3. **Shot** (`run_shot`). For each shot:
   1. Loads the VTA data and the equilibrium (`load_ts_data`). If the equilibrium is unavailable (`EquilibriumUnavailableError`), every interval of the shot is recorded as `skipped` and the run continues.
   2. Maps all TS samples of the shot to rho_pol and midplane R without filters (`preprocess_ts_data` with `apply_filters=False`). This output is the `mapped_data` cache entry.
   3. Runs the shot-wide outlier stage (validity and the local statistical test, `apply_shot_wide_statistical_outlier_filter`). This runs only when `local_statistical_outlier_config` is present and enabled.
   4. Calls `run_interval` for each interval of the shot. A failed interval is recorded and the others still run. Each interval gets a JSON file, and the shot gets `interval_status.csv`.
   5. Saves the power-signal figure when `plots.power_signals` is true.
4. **Interval** (`run_interval`). For one interval:
   1. Selects the samples with `t_begin <= time <= t_end`.
   2. Applies the interval filters (`apply_preprocessing_filters`): rho_pol window, channel trim and window limits.
   3. Applies the density-outside-separatrix, cold-dense-edge and distance outlier filters, in that order.
   4. Sorts the remaining samples by midplane radius (`map_to_r_midplane`).
   5. **Spline stage.** Computes knots with `compute_channel_knots`, unless `spline.R_knots` is given. Fits Te and Ne jointly with `ondrej_spline` or `ipfn_spline`, then evaluates the profiles on a grid with step `dR_fine`.
   6. **mtanh stage.** Fits Te and Ne separately with `fit_modified_tanh_profile`.
   7. **Pedestal stage.** Evaluates both fits at the pedestal radius R_ped, the equilibrium R at `pedestal.rho_pol` (default 0.95).
   8. **Physics stage** (when `physics.enabled` is true, the default, and a connection-length provider is passed). Loads the power-balance inputs (`load_physics_inputs`) and the connection length (`get_connection_length`), and computes the equilibrium separatrix at `physics.rho_sep` (`compute_equilibrium_separatrix`). Then runs the pysepest separatrix estimate (`estimate_separatrix`) on the spline and mtanh profiles, on `physics.grid_size` radial points. The power-balance inputs and the connection length are cached when their caches are enabled.
   9. **Figures** (when `plots.enabled` is true): summary or comparison figures. A failed interval gets a failure diagnostic when `plots.failure_diagnostics` is true.
   10. Builds the interval result: the JSON content (`_interval_json`) and the status summary (`_result_summary`).
5. **Outputs.** `run_shot` writes each shot's `interval_status.csv`. `run_database` writes `run_summary.json`, and the script saves the enriched pickle and CSV. Enrichment (`enrich_database`) is a left join on shot, `t_begin` and `t_end`, so every processed interval keeps its row.

While it runs, the pipeline prints progress for each shot and interval, including cache use and failures. The main functions can also be imported from `ts_profile_fitter.pipeline`: `run_interval` returns one interval's result, `run_shot` processes one shot, and `run_database` processes the whole database.

## Configuration: how the YAML is read

- **Path resolution.** Relative paths are resolved against the folder of the YAML file, so a run works from any working directory. The resolved keys are `shot_list`, `database` (or `intervals`), `data_dir`, `output_dir`, `output_database`, `output_database_csv`, `cache.directory`, `fit_config.plots.summary_directory` and `fit_config.plots.diagnostics_directory`. Absolute paths are kept. `plots.power_directory` is not resolved, so a relative value is taken from the working directory.
- **Section precedence.** The `fit_config` mapping is read first. Each top-level section `preprocess`, `fits`, `physics`, `plots` and `cache`, when present, replaces the same-named section of `fit_config` as a whole; keys are not merged. Top-level `diagnostic`, `equilibrium` and `fit_config_id` override the values in `fit_config`.
- **Unused key.** `data_dir` is passed to `load_ts_data`, which does not use it. TS data come from the AUG VTA shotfile, and the equilibrium comes from the shotfile named by `equilibrium`.
- **Interval table.** `load_intervals` reads `.pkl`, `.csv` or Excel. The required columns are `shot_id` (or `shot`), `t_begin` (or `t_begin (s)`) and `t_end` (or `t_end (s)`), in seconds. `label` (or `profile_type`) is optional. A table that contains both names of one column, such as `shot_id` and `shot`, is rejected. The raw database column names can therefore be used directly.
- **Shot list.** `load_shot_list` reads a CSV or Excel file.
- **Labels.** `fit_config_id` is stored as a label in the database only (default `default`). `equilibrium` (default `EQH`) and `diagnostic` (default `VTA`) select the data sources.
- **Connection-length provider.** The provider function is passed in code (`connection_length_provider=get_LcSOL`), not from the YAML. `cache.provider_id` is only the label used in the connection-length cache key.

## Configurable parameters

Each table gives the key, the stage that reads it, its effect, the thesis value from `configs/pipeline_config_run_06_for_thesis.yaml`, and the code default used when the key is absent. "Off" means the stage is skipped.

### Run, paths and caches (top level)

| Key | Stage | Effect | Thesis run 06 | Code default |
| --- | --- | --- | --- | --- |
| `shot_list` | `run_database` | Shots to process | `shot_list.csv` | `shot_list.csv` |
| `database` (or `intervals`) | `load_intervals` | Interval table; one output row per processed interval | `../data/34383---38808_database_dt-0.1s.pkl` | `intervals.csv` |
| `data_dir` | `load_ts_data` | Passed through and not used | `../data` | `.` |
| `output_dir` | all | Root of all outputs | `../results/EQH_run_06_for_thesis` | `results` |
| `output_database`, `output_database_csv` | `scripts/run_full_database.py` | Enriched database files | `..._fitted.pkl`, `..._fitted.csv` | `enriched_database.pkl`, `enriched_database.csv` |
| `equilibrium` | shot and physics stages | `EQH` or `IDE` | `EQH` | `EQH` |
| `diagnostic` | `load_ts_data` | TS diagnostic | `VTA` | `VTA` |
| `fit_config_id` | database column | Label only | `mtanh` | `default` |
| `cache.directory` | all caches | Cache folder | `../results/EQH_run_06_for_thesis/cache` | `<output_dir>/cache` |
| `cache.reuse_failed` | all caches | Reuse stored failed results | `false` | `false` |
| `cache.provider_id` | connection-length cache | Label in the cache key | `get_LcSOL_selectable_v1` | `get_LcSOL_v1` |
| `cache.mapped_data.enabled` | mapped-data cache | Cache the whole-shot mapping | `true` | off |
| `cache.physics_inputs.enabled` | physics-input cache | Cache the power-balance inputs | `true` | off |
| `cache.connection_length.enabled` | connection-length cache | Cache connection lengths | `true` | off |

### Preprocessing and outlier filters (`fit_config.preprocess`)

| Key | Stage | Effect | Thesis run 06 | Code default |
| --- | --- | --- | --- | --- |
| `mapping_method` | mapping | Backend for the rho_pol mapping | `ipfnpytools` | `ipfnpytools` |
| `r_edge_offset` (alias `r_shift`) | mapping | Radial offset (m) applied to the edge channels | `0.008` | `0.008` |
| `rhop_filter` | shot-wide support; interval | Keeps rows with `rhop_min <= rhop <= rhop_max` in each listed system (`core`, `edge`, or one block for both) | `core` only, `rho_pol` 0.0 to 0.97; the edge is not filtered | off |
| `channel_trim` | shot-wide support; interval | Removes the `first` and `last` channels of each system, ordered by midplane R | off | off |
| `local_statistical_outlier_config` | shot-wide stage | Validity checks and the local statistical test | on: `upper_value_threshold` 5, `relative_error_threshold` 15, `radial_half_width_m` core 0.03 and edge 0.01, `time_half_width_s` 0.025, `min_neighbors` 6, `min_radial_neighbors` 3, `min_temporal_neighbors` 2, `fallback_to_nearest_channel` true | off, and the validity checks are off too |
| `outlier_config` | interval | Window limits per `Te_core`, `Te_edge`, `Ne_core`, `Ne_edge` (or `Te`, `Ne`) | off | off |
| `density_outside_separatrix_config` | interval | Removes Ne above `median_factor` times the interval median, where rho_pol is above `rho_pol_min` | `enabled: false` (`median_factor` 1, `rho_pol_min` 1) | `median_factor` 0.5, `rho_pol_min` 1.0 |
| `cold_dense_edge_config` | interval | Removes edge rows with `Te < te_max` and `Ne > ne_min` | `enabled: false` (`te_max` 1.1, `ne_min` 1e19) | `te_max` 1.1, `ne_min` 2e19 |
| `distance_outlier_config.Ne_edge` | interval | Same-channel temporal outlier test on edge Ne | `enabled: false` | `n_neighbors` 5, `zscore` 8.0 |
| `report_outliers` | all filters | Prints a report for each filter | not set | `false` |

A block that is present is active unless it sets `enabled: false`. The code defaults of the last three filters apply only when their block is present.

### Spline and mtanh fits (`fit_config.fits`)

| Key | Stage | Effect | Thesis run 06 | Code default |
| --- | --- | --- | --- | --- |
| `mtanh.Te.deg_tail`, `mtanh.Ne.deg_tail` | mtanh | Degree (0 to 3) of the tail polynomial | `1`, `1` | `3`, `3` |
| `mtanh.Te.deg_top`, `mtanh.Ne.deg_top` | mtanh | Degree (0 to 3) of the top polynomial | `3`, `3` | `3`, `3` |
| `mtanh.Te.fix_a_tail` | mtanh | Fixes the tail level (parameter index 2) and removes it from the free parameters | `0` | free |
| `mtanh.Te.fix_x0`, `mtanh.Te.fix_w` (and the Ne block) | mtanh | Fixes the centre or the width | not set | free |
| `mtanh.Te.y_scale`, `mtanh.Ne.y_scale` | mtanh | Scale of the quantity used in the fit | `1e3` (Te), `1e19` (Ne) | `1.0` |
| `spline.method` | spline | Solver: `ipfn_spline` or `ondrej_spline` | `ipfn_spline` | `ondrej_spline` |
| `spline.knot_strategy` | `compute_channel_knots` | Knot placement (see below) | `hybrid` | `global_quantile` |
| `spline.knot_count_shift` | `compute_channel_knots` | Added to the knot count; for core-anchored strategies, added to the edge-knot count | `+2` | `0` |
| `spline.core_merge_rhop_spacing_factor` | `compute_channel_knots` | Merges adjacent core channels whose rho_pol gap is below this factor times the median gap (core-anchored strategies) | not set | `0.5` |
| `spline.R_knots` | spline | Explicit knot positions (m); skips `compute_channel_knots` | not set | computed |
| `spline.dR_fine` | spline | Step (m) of the evaluation grid | `0.002` | `5e-4` |
| `spline.dR_channel_max` | `find_x_means` | Knot grouping distance (m). Used only when no knots are given, so it has no effect in the batch pipeline | `0.003` | `3e-3` |
| `spline.ne_scale`, `spline.Te_scale` | both methods | Scales of the optimisation variables | `1e19`, `1000` | `1e19`, `1e3` |
| `spline.s_factor`, `spline.mono`, `spline.maxiter` | `ipfn_spline` | Smoothing factor; monotonicity constraint; iteration limit | `1.0`, `decreasing`, `500` | `1.0`, `decreasing`, `200` |
| `spline.sigma_conv`, `spline.max_step_scale`, `spline.use_errors`, `spline.loss`, `spline.f_scale` | `ondrej_spline` only | Gaussian convolution width (m); step limit; error weighting; loss function; loss scale | `0.0`, `3`, `true`; `loss` and `f_scale` not set. Ignored by `ipfn_spline`, so inert in run 06 | `5e-3`, `3`, `true`, `linear`, `1.0` |

The mtanh model is `y = 0.5 (1 + T) tail(x) + 0.5 (1 - T) top(x)`, with `T = tanh((x - x0) / w)`, and cubic polynomials `tail` and `top` of the configured degrees. Parameter index 2 is the tail level `a_tail`.

Knot strategies (`compute_channel_knots`; N is the number of retained channels and N_edge the number of retained edge channels):

- `global_quantile`: `ceil(N / 2) + knot_count_shift` knots at quantiles of the midplane radii.
- `hybrid`: one knot per core channel group (adjacent core channels are merged when their rho_pol gap is small), plus `ceil(N_edge / 2) + knot_count_shift` knots at quantiles of the per-channel edge radii.
- `edge_biased_quantile`: as `hybrid`, but the edge quantiles use every edge sample, so knots follow where the data are dense.
- `balanced_quantile`: also accepted; see `compute_channel_knots`.

### Physics and pedestal (`fit_config.physics`, `fit_config.pedestal`)

| Key | Stage | Effect | Thesis run 06 | Code default |
| --- | --- | --- | --- | --- |
| `physics.enabled` | physics stage | Runs the power balance, connection length and separatrix estimates; needs a connection-length provider | `true` | `true` |
| `physics.rho_sep` | equilibrium separatrix | rho_pol of the reference separatrix (`Rsep_eqh`) | `1.0` | `1.0` |
| `physics.grid_size` | separatrix model | Number of radial points on which the model is evaluated | `4000` | `2000` |
| `physics.dWdt_method` | `load_physics_inputs` | Source of dW/dt: `fix_to_0` (zero), `paper` (0.25 times P_NET), `tot` (TOT dWmhd/dt), `inter_elm` or `fast_wmhd`. All methods give zero when f_ELM is unavailable | `fix_to_0` | `fix_to_0` (the loader's own default is `paper`) |
| `physics.elm_fraction` | `load_physics_inputs` | ELM-cycle window used by `inter_elm` and `fast_wmhd` | not set | `0.75` to `0.95` |
| `physics.model` | `estimate_separatrix` | Model constants `f_e`, `f_out`, `f_cond`, `q_distribution`, `min_lambda` | not set | `0.8`, `0.65`, `0.8`, `1.0`, `1e-8` |
| `physics.connection_length.dR` | `get_connection_length` | Step (m) of the connection-length integration | not set | `1e-3` |
| `physics.connection_length.plot` | `get_connection_length` | Passes a plot flag to the provider | not set | `false` |
| `pedestal.rho_pol` | pedestal stage | rho_pol of the pedestal radius R_ped | not set | `0.95` |

### Plots (`fit_config.plots`)

| Key | Stage | Effect | Thesis run 06 | Code default |
| --- | --- | --- | --- | --- |
| `enabled` | interval | Saves figures | `true` | `false` |
| `kind` | interval figures | `summary`: one PDF per interval in `profile_summary/shot_<id>/`. `comparison`: comparison figures in the shot folder | `summary` | `comparison` |
| `r_zoom`, `r_zoom_max` | comparison figures | Adds a zoomed comparison figure when both are set | not set | not set |
| `summary_directory`, `diagnostics_directory` | figures | Output roots, resolved against the YAML folder | `../results/EQH_run_06_for_thesis/profile_summary` and `.../profile_diagnostics` | `output_dir` |
| `failure_diagnostics` | `_plot_failure_diagnostic` | Diagnostic figure for each failed interval | `true` | `false` |
| `power_signals` | `run_shot` | Power-signal figure per shot | `false` | `false` |
| `power_directory` | `run_shot` | Folder for power-signal figures; not resolved against the YAML folder | not set | shot output folder |

## Outlier filtering

Filters act on rows: one row is one channel at one time. A row removed by any filter is removed for both Te and Ne. Removed rows are not written to the database or to the interval JSON files. The steps run in the order below. The thesis configuration uses steps 1 to 3 only.

Shot-wide stage (once per shot, before the shot is cut into intervals; runs only when `local_statistical_outlier_config` is present and enabled):

1. **Validity.** Removes rows where Te or Ne is non-finite, zero or negative, or where its error is non-finite, zero or negative. Labels: `Te_nonfinite_value`, `Ne_zero_error`, and so on.
2. **Local statistical test.** Run separately for Te and Ne, for each system (core or edge) and each shot.
   - Neighbourhood: the other valid samples of the same system and shot within `radial_half_width_m[system]` in midplane R and `time_half_width_s` in time.
   - Support: a sample is left unclassified when it has fewer than `min_neighbors` valid neighbours, fewer than `min_radial_neighbors` distinct channels (its own channel counts), or fewer than `min_temporal_neighbors` distinct times. With `fallback_to_nearest_channel: true`, the nearest channels by rho_pol within the same time window are added until the radial minimum is met.
   - Test: a sample is removed when its value, or its relative error |error / value|, is more than `upper_value_threshold` (or `relative_error_threshold`) robust standard deviations above the neighbourhood median. The robust standard deviation is 1.4826 times the median absolute deviation (MAD). The test is one-sided and rejects high values only. A threshold that is not given falls back to `threshold` (default 3.5).
   - Labels: `Te_statistical_upper_value_outlier`, `Te_statistical_relative_error_outlier`, and the Ne equivalents.
   - Scope: the rho_pol window and the channel trim decide which samples act as neighbours and which are tested. Samples outside them are not removed by this stage. The test runs on the whole shot, so the neighbourhood does not change at interval edges.

Interval stage (in `run_interval`, after the shot is cut into intervals):

3. **rho_pol window** (`rhop_filter`). Removes rows outside `[rhop_min, rhop_max]` in each listed system. Label: `rhop_filter`.
4. **Channel trim** (`channel_trim`). Removes the first and last channels of each system. Label: `channel_trim`.
5. **Window limits** (`outlier_config`). Per quantity and system, removes rows with values outside `value_min` to `value_max`, errors outside `error_min` to `error_max`, relative error above `rel_error_max`, or zeros when `remove_zeros` is true.
6. **Density outside separatrix.** Removes rows with Ne above `median_factor` times the interval median Ne, where rho_pol is above `rho_pol_min`. Label: `high_density_outside_separatrix`.
7. **Cold-dense edge.** Removes edge rows with Te below `te_max` and Ne above `ne_min`. Label: `cold_dense_edge_outlier`.
8. **Distance outlier** (`Ne_edge`). Each edge Ne sample is compared with its `n_neighbors` nearest samples of the same channel, ordered by time. It is removed when it differs from their median by more than `zscore` times the robust scale (1.4826 times MAD, and at least `min_scale`), and it passes `min_value` and `max_relative_error`. For Ne, when the group has at most `short_group_max_points` samples and the same-sample Te is at most `te_floor` with relative error at most `te_max_relative_error`, the neighbour minimum replaces the median as the reference. Label: `Ne_distance_outlier`.

## Run database

The thesis database is `results/EQH_run_06_for_thesis/34383---38808_database_dt-0.1s_fitted.pkl`. It is tracked in git and has 752 rows, 512 columns, 14 shots and a size of 3,050,547 bytes. The CSV with the same content is git-ignored and is written by the same run. Read the pickle with pandas alone:

```python
import pandas as pd

df = pd.read_pickle("results/EQH_run_06_for_thesis/34383---38808_database_dt-0.1s_fitted.pkl")
```

Each row is one processed interval of the input database, keyed by `shot`, `t_begin (s)` and `t_end (s)`. The input database `data/34383---38808_database_dt-0.1s.pkl` is git-ignored and not in the repository; the tracked pickle already contains its columns.

### Columns

| Group | Columns | Notes |
| --- | --- | --- |
| Keys and labels | `shot`, `t_begin (s)`, `t_end (s)`, `label`, `label_result`, `equilibrium`, `diagnostic`, `fit_config_id` | `label_result` is the label written by the fit |
| Status | `fit_status`, `fit_error`, `failed_stage`, `spline_status`, `mtanh_status`, `physics_status` | See the status table below |
| Sizes | `n_points`, `n_knots` | `n_points` is the number of rows entering the spline fit |
| Fit metrics | `spline_Te_sse`, `spline_Te_weighted_sse`, `spline_Te_reduced_chi2`, the same for Ne, and the `mtanh_` set | From `fit_metrics.py` |
| Separatrix | `spline_` and `mtanh_` prefixes on `Rsep`, `Te_sep`, `Ne_sep`, `P_SOL`, `P_e_SOL`, `p_sep`, `lambda_p_sep`, `lambda_Ne_sep`, `lambda_Te_sep`, `lambda_q_sep`, `q_par_P_SOL_sep`, `q_par_SH_sep`; plus `Rsep_eqh` | `Rsep` in m, `Te_sep` in eV, `Ne_sep` in m^-3. `Rsep_eqh` is the equilibrium separatrix at `physics.rho_sep` |
| Pedestal | `spline_` and `mtanh_` prefixes on `Te_ped`, `Ne_ped`, `p_ped`, `lambda_Te_ped`, `lambda_Ne_ped`, `lambda_p_ped`; plus `Rped_eqh` | Values at R_ped. `p_ped` is in kPa (Ne times Te times the elementary charge, divided by 1000). `Rped_eqh` is the equilibrium R at `pedestal.rho_pol` |

The 446 columns named `GROUP|name` are input signals copied from the input database, for example the `BPT`, `GQH`, `MAC` and `TOT` groups. Of these, 302 have an empty group prefix and are not characterised here.

### Status values

Counts for run 06 (752 rows):

| Column | Values | Meaning |
| --- | --- | --- |
| `fit_status` | `ok` 710, `failed` 42 | `ok` includes warnings. `skipped` is used when the equilibrium is unavailable; none occur in run 06 |
| `failed_stage` | empty 710, `spline` 15, `power_balance` 14, `preprocess` 12, `mtanh` 1 | First stage that failed; later stages are `not_run` |
| `spline_status` | `ok` 725, `failed` 15, `not_run` 12 | |
| `mtanh_status` | `ok` 724, `failed` 1, `not_run` 27 | |
| `physics_status` | `ok` 710, `failed` 14, `not_run` 28 | `power_balance` failures appear here |

A `preprocess` failure makes all three stage statuses `not_run`. A `spline` failure makes `mtanh_status` and `physics_status` `not_run`.

### Per-interval files

Each processed interval has `shot_<id>/interval_<t_begin>_<t_end>.json`, with the times written to three decimals. It holds the fitted parameters and their standard errors, the spline knots and coefficients, the fit metrics, and the scalar physics results. It does not hold covariance matrices or physics grids. Each shot folder also has `interval_status.csv`, with one row per interval.

## Caching

Three disk caches exist. Each is enabled separately under `cache` (`mapped_data`, `physics_inputs`, `connection_length`) and is off unless enabled. Files are named `<kind>_<sha256 prefix>.pkl` under `cache.directory` and are written atomically.

| Cache | Key fields | Notes |
| --- | --- | --- |
| `mapped_data_v1` | shot, diagnostic, equilibrium, `mapping_method`, `r_edge_offset` | Stores the whole-shot mapping before any filter, so changing filters or thresholds does not invalidate it |
| `physics_inputs_v2` (files `physics_*.pkl`) | shot, `t_begin`, `t_end`, equilibrium | Does not include `dWdt_method` or `elm_fraction` (set in the YAML), or the BPT and TTH experiment names (function defaults `bpt_experiment='DAVIDP'`, `tth_experiment=None`). After changing these, delete `cache/physics_*.pkl` or bump the version |
| `connection_length_v1` | provider label, equilibrium, shot, interval midpoint time, `dR` | Changing the provider label makes existing entries unreachable |

- Only entries with `status: ok` are reused, unless `cache.reuse_failed` is true.
- Only plain data is stored, not shotfile objects. The source database is not modified.
- Each newly computed physics input and connection length is appended to `db_Lgil2024_physics_input.csv` and `db_Lgil2024_connection length.csv` in `output_dir`. These are reference tables. Rows are appended without de-duplication, so repeated runs add duplicates.

## Outputs and figures

A run writes the following under `output_dir`:

```text
<output_dir>/
  shot_<id>/                          one folder per shot
    interval_<t_begin>_<t_end>.json
    interval_status.csv
  cache/                              when caching is enabled
  profile_summary/shot_<id>/          summary figures (plots.kind: summary)
  profile_diagnostics/shot_<id>/      diagnostics for failed intervals
  run_summary.json
  <database>_fitted.pkl
  <database>_fitted.csv
  db_Lgil2024_physics_input.csv       when new physics inputs are computed
  db_Lgil2024_connection length.csv   when new connection lengths are computed
```

The scripts listed under Auxiliary scripts add `spline_knot_scan_<timestamp>.csv` and `scaling_laws_separatrix/`.

Figure names follow `shot{id}_{EQ}_k{NN:02d}_t{t0}-{t1}_{kind}_{strategy}.pdf`. Here `k` is the knot count, `p` replaces the decimal point, `kind` is `summary` or a comparison name, and `strategy` is the knot strategy. Example: `shot35970_EQH_k17_t3p483-3p509_summary_hybrid.pdf`. Comparison figures use names such as `te_comparison_full`, and `te_comparison_zoom` when `r_zoom` and `r_zoom_max` are both set.





## Auxiliary scripts

- `scripts/initialize_run.py`: creates a run-specific configuration (see Quick start).
- `scripts/scan_spline_knots.py --config FILE --shot-id ID [ID ...] --knot-shifts K [K ...] [--time T] [--strategies S ...] [--output CSV]`: scans the knot strategies (`global_quantile`, `balanced_quantile`, `edge_biased_quantile`, `hybrid`) and the `knot_count_shift` values for the given shots, using the configured caches. `--time` restricts the scan to the interval containing T. The output is one CSV, by default `spline_knot_scan_<timestamp>.csv` in `output_dir`. It needs AUG data access and `aug_separatrix/`.
- `scripts/separatrix_scaling_laws.py [--input CSV] [--output-dir DIR] [--shots notebook|all|IDs] [--shapiro-threshold 0.03]`: fits the power law `ln y = gamma_P ln P + gamma_G ln Gamma_D + gamma_I ln I_p + ln k` to the spline separatrix and pedestal-top quantities, with a leave-one-shot-out jackknife. `--shots notebook` (the default) uses the 14 shots pinned in the script. Output goes to `results/EQH_run_06_for_thesis/scaling_laws_separatrix/` by default. The default input is the fitted CSV, which is git-ignored, so run the pipeline first or pass `--input`.

Notebooks in `notebooks/` include `single_shot_and_interval.ipynb` (inspects one interval and compares it with batch results), `pedestal_core_fits.ipynb` (the analysis that the scaling-law script replicates), and `spline_knot_scan_analysis.ipynb`.

## Plotting

Figures use Matplotlib. `plotting.py` imports the `ipfnpytools.plot.plots` and `ipfnpytools.save_figure` helpers in a guarded block. Without `ipfnpytools`, the plotting functions raise `ImportError` when called, and saving raises it too. Figures are saved as PDF. Plots are off by default; the thesis run enables them, so it needs `ipfnpytools`.

## Validation

Run the tests from the repository root in the active conda environment:

```bash
python -m pytest -q
```

`pytest.ini` sets `testpaths = tests` and adds `.` and `src/` to the import path. `tests/test_pipeline.py` imports helpers from `scripts/scan_spline_knots.py`, which loads `get_LcSOL` from `../aug_separatrix/` at import time.

Current result: 57 tests collected, 9 failed, 48 passed (3.8 s).

| File | Failing tests |
| --- | --- |
| `tests/test_fitting.py` | `test_fit_modified_tanh_profile_returns_finite_params`, `test_fit_ipfn_spline_ne_Te_TS_returns_shared_profile_outputs`, `test_fit_ipfn_spline_preserves_interior_endpoint_strategy_knots` |
| `tests/test_plotting.py` | `test_filtered_profile_plot_colors_by_time_and_overlays_outliers`, `test_filtered_profile_plot_annotates_fit_metrics_per_axis`, `test_spline_profile_plot_combines_te_ne_with_shared_title_and_metrics` |
| `tests/test_preprocess.py` | `test_local_statistical_outlier_filter_uses_2d_support_and_both_criteria`, `test_local_statistical_outlier_filter_falls_back_to_nearest_channel`, `test_ipfnpytools_mapping_returns_finite_2d_arrays` |

Two of the failures are local statistical filter tests. Until they pass, treat the behaviour described in Outlier filtering as verified by code reading only. The causes of the failures are not analysed here.

## Known limitations

- `load_physics_inputs` overwrites `Z_eff` with 1.5 after reading the `Zeff` signal (`loading.py`, line 644).
- dW/dt is set to zero for every `dWdt_method` when f_ELM is unavailable.
- `ipfn_spline` ignores `sigma_conv`, `max_step_scale`, `use_errors`, `loss` and `f_scale`, so the thesis values for these keys have no effect.
- `dR_channel_max` has no effect in the batch pipeline, which always supplies knots.
- The code default spline method is `ondrej_spline`; both committed configurations set `ipfn_spline`.
- The physics-input cache key omits `dWdt_method` and `elm_fraction` (see Caching).
- The `db_Lgil2024_*.csv` reference tables append duplicate rows on repeated runs.
- The committed template sets `cache.provider_id: get_LcSOL_v1`, while the thesis file uses `get_LcSOL_selectable_v1`. Changing the label makes existing connection-length cache entries unreachable.
- `plots.power_directory` is not resolved against the YAML folder.
- `initialize_run.py` writes plot paths under `fit_config.plots`; a top-level `plots` section would override them.
- `data_dir` is accepted but not used.
- `separatrix_scaling_laws.py` reads a git-ignored CSV by default.
