#!/usr/bin/env python
"""Separatrix and pedestal-top scaling laws from the spline fits of a database run.

Replicates the 4D power-law fit and leave-one-shot-out jackknife of
notebooks/pedestal_core_fits.ipynb (cells 12, 16 and 18) for the spline
separatrix and pedestal-top quantities:

    ln y = gamma_P ln P + gamma_G ln Gamma_D + gamma_I ln I_p + ln k

Selection follows pinned cells 6 and 12 restricted to developed EDA labels,
1 - MAC|shapiro(Ipolsola) < 0.03, notebook shot list. Shots are weighted by
1/N_shot and r^2 is the weighted Pearson correlation between the scaled
variable and y, as in cell 16.

Jackknife SEs use sqrt((n-1)/n * sum((theta_i - mean)^2)) over the n
leave-one-shot-out fits. Pinned cell 18 indexes [:-1] on both pseudo-value
terms, which makes its "Standard Deviation" the sample SD of the leave-one-out
fits (about sqrt(n)/(n-1) of the jackknife SE) and its "Standard Error"
(n-1) times too small.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / 'results' / 'EQH_run_06_for_thesis'
    / '34383---38808_database_dt-0.1s_fitted.csv')
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / 'results' / 'EQH_run_06_for_thesis' / 'scaling_laws_separatrix')

# Pinned cell 12 shot list (36257 is commented out there).
NOTEBOOK_SHOTS = (
    36157, 36124, 37193, 36349, 36151, 38049, 38066, 38067,
    38742, 38743, 38744, 36980, 35970, 38808)

SHAPIRO_COLUMN = 'MAC|shapiro(Ipolsola)'
GAS_COLUMN = 'UVS|D_tot'
GAS_FACTOR = 1e-21  # e/s -> 1e21 e/s
CURRENT_COLUMN = 'FPC|IpiFP'
CURRENT_FACTOR = 1e-6  # A -> MA
POWER_FACTOR = 1e-6  # W -> MW

POWER_VARIANTS = (
    # key, CSV column, symbol, gamma symbol
    ('Ptot', 'TOT|P_TOT', r'P_{\mathrm{tot}}', r'\gamma_{P_{\mathrm{tot}}}'),
    ('Psep', '|P_TOT-Pr_sep', r'P_{\mathrm{SOL}}', r'\gamma_{P_{\mathrm{SOL}}}'),
)
# LaTeX symbols for the tables; POWER_VARIANTS symbols are for the plots.
LATEX_POWER = {
    'Ptot': (r'\Ptot{}', r'\gamma_{\Ptot{}}'),
    'Psep': (r'\PSOL{}', r'\gamma_{\PSOL{}}'),
}

SEPARATRIX_TARGETS = (
    # key, CSV column, factor to display units, plot label
    ('n_e', 'spline_Ne_sep', 1e-19, r'$n_{e,sep,spl}$ ($10^{19}$ m$^{-3}$)'),
    ('T_e', 'spline_Te_sep', 1.0, r'$T_{e,sep,spl}$ (eV)'),
    ('p_e', 'spline_p_sep', 1.0, r'$p_{e,sep,spl}$ (kPa)'),
    ('lambda_Te', 'spline_lambda_Te_sep', 1e3, r'$\lambda_{T_{e},spl}$ (mm)'),
    ('lambda_Ne', 'spline_lambda_Ne_sep', 1e3, r'$\lambda_{n_{e},spl}$ (mm)'),
    ('lambda_p', 'spline_lambda_p_sep', 1e3, r'$\lambda_{p_{e},spl}$ (mm)'),
)
PEDESTAL_TARGETS = (
    ('n_e_ped', 'spline_Ne_ped', 1e-19, r'$n_{e,ped,spl}$ ($10^{19}$ m$^{-3}$)'),
    ('T_e_ped', 'spline_Te_ped', 1.0, r'$T_{e,ped,spl}$ (eV)'),
    ('p_e_ped', 'spline_p_ped', 1.0, r'$p_{e,ped,spl}$ (kPa)'),
)
TARGETS = SEPARATRIX_TARGETS + PEDESTAL_TARGETS
TABLE_LABELS = {
    'n_e': r'\Nesep{}',
    'T_e': r'\Tesep{}',
    'p_e': r'\Pesep{}',
    'lambda_Te': r'\lamTe{}',
    'lambda_Ne': r'\lamNe{}',
    'lambda_p': r'\lamPe{}',
    'n_e_ped': r'\Neped{}',
    'T_e_ped': r'\Teped{}',
    'p_e_ped': r'\Peped{}',
}
TABLE_GROUPS = (
    # file tag, targets, short caption, description for the long caption
    ('separatrix', SEPARATRIX_TARGETS, 'Separatrix scaling exponents',
     'separatrix parameters'),
    ('pedestal', PEDESTAL_TARGETS, 'Pedestal-top scaling exponents',
     'pedestal-top parameters'),
)

REGIMES = {
    # Same colour and marker as the scatter notebook palette (Paired[5]).
    'developed EDA H-mode': {'color': matplotlib.colormaps['Paired'](5), 'marker': 'x'},
}


def regime_of(label):
    """Pinned cell 6 developed EDA label; ELMy and early labels return None."""
    text = str(label).lower()
    if 'developed' in text and 'eda' in text and 'elmy' not in text:
        return 'developed EDA H-mode'
    return None


def select_sample(df, shots=NOTEBOOK_SHOTS, shapiro_threshold=0.03):
    regimes = df['label'].map(regime_of)
    keep = (
        (df['physics_status'] == 'ok')
        & regimes.notna()
        & ((1 - df[SHAPIRO_COLUMN]) < shapiro_threshold))
    if shots is not None:
        keep &= df['shot'].isin(shots)
    sample = df.loc[keep].copy()
    sample['regime'] = regimes[keep]
    return sample


def build_frame(sample, power_column, target_column, target_factor):
    frame = pd.DataFrame({
        'shot': sample['shot'].to_numpy(),
        'regime': sample['regime'].to_numpy(),
        'P': sample[power_column].to_numpy(dtype=float) * POWER_FACTOR,
        'G': sample[GAS_COLUMN].to_numpy(dtype=float) * GAS_FACTOR,
        'I': sample[CURRENT_COLUMN].to_numpy(dtype=float) * CURRENT_FACTOR,
        'Y': sample[target_column].to_numpy(dtype=float) * target_factor,
    })
    values = frame[['P', 'G', 'I', 'Y']].to_numpy()
    valid = np.all(np.isfinite(values) & (values > 0), axis=1)
    return frame.loc[valid].reset_index(drop=True)


def _shot_weights(shots):
    """Return 1/N_shot per row, i.e. sigma^-2 with sigma = sqrt(N_shot)."""
    _, inverse, counts = np.unique(shots, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _weighted_corr(x, y, w):
    def mean(a):
        return np.sum(w * a) / np.sum(w)

    def cov(a, b):
        return np.sum(w * (a - mean(a)) * (b - mean(b))) / np.sum(w)

    return cov(x, y) / np.sqrt(cov(x, x) * cov(y, y))


def _estimate(frame):
    """Return [gamma_P, gamma_G, gamma_I, ln k, r^2] of the shot-weighted fit.

    Weighted least squares on ln Y with weights 1/N_shot; same optimum as
    curve_fit with sigma = sqrt(N_shot) in pinned cell 16.
    """
    design = np.column_stack([
        np.log(frame['P']), np.log(frame['G']), np.log(frame['I']),
        np.ones(len(frame))])
    w = _shot_weights(frame['shot'].to_numpy())
    sqrt_w = np.sqrt(w)
    gamma_P, gamma_G, gamma_I, ln_k = np.linalg.lstsq(
        design * sqrt_w[:, None], np.log(frame['Y']).to_numpy() * sqrt_w,
        rcond=None)[0]
    scaled = frame['P'] ** gamma_P * frame['G'] ** gamma_G * frame['I'] ** gamma_I
    r2 = _weighted_corr(scaled.to_numpy(), frame['Y'].to_numpy(), w) ** 2
    return np.array([gamma_P, gamma_G, gamma_I, ln_k, r2])


def fit_scaling_law(frame):
    """Return full-sample estimates, leave-one-shot-out jackknife SEs and n shots."""
    shots = frame['shot'].unique()
    estimate = _estimate(frame)
    loo = np.array([_estimate(frame[frame['shot'] != shot]) for shot in shots])
    n = len(shots)
    se = np.sqrt((n - 1) / n * np.sum((loo - loo.mean(axis=0)) ** 2, axis=0))
    return estimate, se, n


def run_fits(sample):
    records, fits = [], {}
    for power_key, power_column, _, _ in POWER_VARIANTS:
        for target_key, target_column, factor, _ in TARGETS:
            frame = build_frame(sample, power_column, target_column, factor)
            estimate, se, n_shots = fit_scaling_law(frame)
            fits[(power_key, target_key)] = (frame, estimate)
            records.append({
                'power': power_key,
                'target': target_key,
                'column': target_column,
                'n_points': len(frame),
                'n_shots': n_shots,
                'gamma_P': estimate[0], 'gamma_P_se': se[0],
                'gamma_Gamma': estimate[1], 'gamma_Gamma_se': se[1],
                'gamma_Ip': estimate[2], 'gamma_Ip_se': se[2],
                'ln_k': estimate[3], 'ln_k_se': se[3],
                'k': np.exp(estimate[3]),
                'r2': estimate[4], 'r2_se': se[4],
            })
    return pd.DataFrame(records), fits


def _value_uncertainty(value, uncertainty):
    """Format value +- uncertainty, with the uncertainty to one significant digit."""
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return f'${value:.2f}$'
    decimals = max(-int(np.floor(np.log10(uncertainty))), 0)
    # Adding 0.0 turns a negative zero left by rounding into 0.0.
    value = round(float(value), decimals) + 0.0
    return f'${value:.{decimals}f} \\pm {uncertainty:.{decimals}f}$'


def write_latex_table(records, power_key, symbol, gamma_symbol, group, path):
    tag, targets, short_caption, description = group
    rows = records[records['power'] == power_key]
    header = (
        r'Quantity & $' + gamma_symbol + r'$ & $\gamma_{\GasPuff{}}$ & '
        r'$\gamma_{\Ip{}}$ & $r^{2}$ \\')
    body = []
    for target_key, _, _, _ in targets:
        row = rows[rows['target'] == target_key].iloc[0]
        cells = [
            TABLE_LABELS[target_key],
            _value_uncertainty(row['gamma_P'], row['gamma_P_se']),
            _value_uncertainty(row['gamma_Gamma'], row['gamma_Gamma_se']),
            _value_uncertainty(row['gamma_Ip'], row['gamma_Ip_se']),
            f'${row["r2"]:.2f}$',
        ]
        body.append('        ' + ' & '.join(cells) + r' \\')
    caption = (
        r'Scaling exponents of the ' + description
        + r' in developed \EDA{} H-mode, against ' + symbol
        + r', \GasPuff{} and \Ip{}. Values are exponent $\pm$ jackknife standard '
        r'error; $r^{2}$ is the squared shot-weighted correlation of the fit.')
    lines = [
        '% Generated by scripts/separatrix_scaling_laws.py',
        r'\includetable{' + short_caption + '}{' + caption + '}{',
        r'    \label{tab:scaling-' + tag + '-' + power_key + '}',
        r'    \small',
        r'    \begin{tabular}{@{}lcccc@{}}',
        r'        \toprule',
        '        ' + header,
        r'        \midrule',
        *body,
        r'        \bottomrule',
        r'    \end{tabular}',
        '}',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_scaling_laws(fits, output_stem):
    n_rows, n_cols = len(TARGETS), len(POWER_VARIANTS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols + 0.6, 2.2 * n_rows + 0.9),
        sharey='row', squeeze=False)
    for col, (power_key, _, symbol, _) in enumerate(POWER_VARIANTS):
        for row, (target_key, _, _, target_label) in enumerate(TARGETS):
            ax = axes[row, col]
            frame, estimate = fits[(power_key, target_key)]
            gamma_P, gamma_G, gamma_I, ln_k, r2 = estimate
            scaled = frame['P'] ** gamma_P * frame['G'] ** gamma_G * frame['I'] ** gamma_I
            for regime, style in REGIMES.items():
                mask = (frame['regime'] == regime).to_numpy()
                ax.scatter(
                    scaled[mask], frame['Y'][mask], s=8, marker=style['marker'],
                    color=style['color'], linewidths=0.7, alpha=0.7)
            k = np.exp(ln_k)
            s_line = np.geomspace(scaled.min(), scaled.max(), 50)
            ax.plot(s_line, k * s_line, color='black', ls='--', lw=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FormatStrFormatter('%.4g'))
                axis.set_minor_formatter(FormatStrFormatter('%.4g'))
            ax.tick_params(labelsize=7)
            ax.text(
                0.04, 0.96, '$k$ = {:.3g}\n$r^2$ = {:.2f}'.format(k, r2),
                transform=ax.transAxes, va='top', ha='left', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.8, edgecolor='none'))
            ax.set_xlabel(
                r'${}^{{{:.2f}}}\,\Gamma_{{D}}^{{{:.2f}}}\,I_{{p}}^{{{:.2f}}}$'.format(
                    symbol, gamma_P, gamma_G, gamma_I),
                fontsize=8)
            if col == 0:
                ax.set_ylabel(target_label, fontsize=8)
            if row == 0:
                ax.set_title('$' + symbol + '$ scaling', fontsize=9)
    handles = [
        Line2D([], [], ls='none', marker=style['marker'], color=style['color'], label=regime)
        for regime, style in REGIMES.items()]
    handles.append(Line2D([], [], ls='--', color='black', lw=1, label=r'power law $k\,s$'))
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for suffix in ('.pdf', '.png'):
        fig.savefig(output_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


def _resolve_path(value, default):
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_shots(value):
    if value == 'notebook':
        return NOTEBOOK_SHOTS
    if value == 'all':
        return None
    return tuple(int(shot) for shot in value.split(',') if shot.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Separatrix scaling laws for spline fits.')
    parser.add_argument('--input', help='fitted database CSV (default: run 06 fitted CSV)')
    parser.add_argument(
        '--output-dir',
        help='output folder (default: results/EQH_run_06_for_thesis/scaling_laws_separatrix)')
    parser.add_argument(
        '--shots', default='notebook',
        help="'notebook' (pinned cell 12), 'all', or comma-separated shot numbers")
    parser.add_argument(
        '--shapiro-threshold', type=float, default=0.03,
        help='keep rows with 1 - MAC|shapiro(Ipolsola) below this value')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = _resolve_path(args.input, DEFAULT_INPUT)
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)

    df = pd.read_csv(input_path, low_memory=False)
    sample = select_sample(df, _parse_shots(args.shots), args.shapiro_threshold)
    if sample.empty:
        raise SystemExit('No rows selected.')
    print(f'Selected {len(sample)} rows from shots {sorted(sample["shot"].unique().tolist())}')
    print(sample.groupby('regime')['shot'].agg(['size', 'nunique']).to_string())

    records, fits = run_fits(sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / 'scaling_laws_separatrix_fits.csv', index=False)
    for power_key, _, _, _ in POWER_VARIANTS:
        latex_symbol, latex_gamma = LATEX_POWER[power_key]
        for group in TABLE_GROUPS:
            path = output_dir / f'scaling_laws_{group[0]}_{power_key}.tex'
            write_latex_table(records, power_key, latex_symbol, latex_gamma, group, path)
    plot_scaling_laws(fits, output_dir / 'scaling_laws_separatrix')

    print(records.round(4).to_string(index=False))
    print(f'Wrote outputs to {output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
