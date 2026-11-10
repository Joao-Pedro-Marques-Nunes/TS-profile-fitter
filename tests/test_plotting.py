import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ts_profile_fitter import plotting
from ts_profile_fitter.plotting import (
    PROFILE_RADIUS_LABEL,
    plot_filtered_profile_with_outliers,
    plot_profile_fit,
)


def test_filtered_profile_plot_colors_by_time_and_overlays_outliers(monkeypatch):
    filtered = pd.DataFrame({
        'system': ['core', 'edge'],
        'r_midplane': [2.1, 2.2],
        'time': [0.9, 1.2],
        'Te': [100.0, 80.0],
        'Te_err': [10.0, 8.0],
    })
    outliers = pd.DataFrame({
        'system': ['core', 'edge', 'edge', 'core', 'edge'],
        'r_midplane': [2.11, 2.21, 2.22, 2.12, 2.23],
        'time': [1.0, 1.1, 1.2, 1.3, 1.4],
        'Te': [0.0, 300.0, 250.0, 150.0, 180.0],
        'Te_err': [1.0, 20.0, 18.0, 15.0, 16.0],
        'filter_reason': [
            'Te_outlier; Te_zero_value; Te_zero_error',
            'Te_statistical_upper_value_outlier',
            'Ne_statistical_relative_error_outlier',
            'rhop_filter',
            '',
        ],
    })
    captured = {}

    def fake_ipfn_plots(**kwargs):
        captured.update(kwargs)
        fig, ax = plt.subplots()
        return fig, ax

    monkeypatch.setattr(plotting, 'ipfn_plots', fake_ipfn_plots)

    fig, ax = plot_filtered_profile_with_outliers(
        filtered,
        outliers,
        'Te',
        show=False,
        nshot=123,
        interval={'t_begin': 0.9, 't_end': 1.2},
        equilibrium='EQH',
    )

    assert captured['x_labels'] == PROFILE_RADIUS_LABEL
    assert captured['y_labels'] == 'Te [eV]'
    assert captured['z_labels'] == 'Time'
    assert captured['z_units'] == 's'
    assert captured['title'] == 'Shot 123, [0.900, 1.200] s, EQH'
    assert captured['labels'] == [
        None, 'Core TS', None, 'Edge TS', 'Other', 'Negative or zero',
        'core channel removal', 'Te_stat', 'Ne_stat']
    assert captured['z'][1].tolist() == [0.9]
    assert captured['z'][3].tolist() == [1.2]
    assert captured['z'][-5:] == [None, None, None, None, None]
    assert captured['plot_kw'][1]['cmap'] == 'plasma'
    assert captured['plot_kw'][3]['cmap'] == 'plasma'
    assert [plot_kw['color'] for plot_kw in captured['plot_kw'][-5:]] == [
        '0.25', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue']
    assert [plot_kw['alpha'] for plot_kw in captured['plot_kw'][-5:]] == [
        0.28] * 5
    assert all('elinewidth' not in plot_kw for plot_kw in captured['plot_kw'][-5:])
    assert captured['markers'][-5:] == ['x', 'D', 'P', '^', 's']
    assert captured['markersizes'] == [0, 9, 0, 9, 6, 6, 6, 6, 6]
    plt.close(fig)


def test_filtered_profile_plot_places_te_and_ne_in_separate_boxes(monkeypatch):
    filtered = pd.DataFrame({
        'system': ['core', 'edge'],
        'r_midplane': [2.1, 2.2],
        'time': [0.9, 1.2],
        'Te': [100.0, 80.0],
        'Te_err': [10.0, 8.0],
        'Ne': [1.0e19, 8.0e18],
        'Ne_err': [1.0e18, 8.0e17],
    })
    outliers = pd.DataFrame({
        'system': ['core', 'edge'],
        'r_midplane': [2.11, 2.21],
        'time': [1.0, 1.1],
        'Te': [0.0, 300.0],
        'Te_err': [1.0, 20.0],
        'Ne': [0.0, 3.0e19],
        'Ne_err': [1.0, 2.0e18],
        'filter_reason': [
            'Te_statistical_upper_value_outlier',
            'Ne_statistical_relative_error_outlier',
        ],
    })
    captured = {}

    def fake_ipfn_plots(**kwargs):
        captured.update(kwargs)
        fig, ax = plt.subplots()
        return fig, ax

    monkeypatch.setattr(plotting, 'ipfn_plots', fake_ipfn_plots)

    fig, ax = plot_filtered_profile_with_outliers(
        filtered,
        outliers,
        ('Te', 'Ne'),
        show=False,
        y_mins_boxes=[-30, -1e18],
        y_maxs_boxes=[1000, 0.4e20],
    )

    assert captured['boxes'] == [0] * 6 + [1] * 6
    assert captured['n_boxes'] == 2
    assert captured['y_labels'] == ['Te [eV]', 'Ne [m$^{-3}$]']
    assert captured['y'][1].tolist() == [100.0]
    assert captured['y'][7].tolist() == [1.0e19]
    assert captured['y_mins_boxes'] == [-30, -1e18]
    assert captured['y_maxs_boxes'] == [1000, 0.4e20]
    plt.close(fig)


def test_filtered_profile_plot_annotates_fit_metrics_per_axis(monkeypatch):
    filtered = pd.DataFrame({
        'system': ['core', 'edge'],
        'r_midplane': [2.1, 2.2],
        'time': [0.9, 1.2],
        'Te': [100.0, 80.0],
        'Te_err': [10.0, 8.0],
        'Ne': [1.0e19, 8.0e18],
        'Ne_err': [1.0e18, 8.0e17],
    })
    captured = {}
    fig, axes = plt.subplots(1, 2)

    def fake_ipfn_plots(**kwargs):
        captured.update(kwargs)
        return fig, axes

    monkeypatch.setattr(plotting, 'ipfn_plots', fake_ipfn_plots)

    fig, ax = plot_filtered_profile_with_outliers(
        filtered,
        None,
        ('Te', 'Ne'),
        show_outliers=False,
        show_fit=True,
        popt={'Te': [100.0], 'Ne': [1.0e19]},
        fit_metrics={
            'Te': {'reduced_chi2': 1.25, 'weighted_sse': 3.75},
            'Ne': {'reduced_chi2': 2.5, 'weighted_sse': 7.5},
        },
        fit_metrics_loc={'Te': 'upper left', 'Ne': 'lower right'},
        show=False,
        nshot=123,
        interval={'t_begin': 0.9, 't_end': 1.2},
        equilibrium='EQH',
    )

    assert captured['boxes'] == [0] * 5 + [1] * 5
    assert captured['labels'] == [
        None, 'Core TS', None, 'Edge TS', 'Modified-tanh fit',
        None, 'Core TS', None, 'Edge TS', 'Modified-tanh fit',
    ]
    assert captured['title'] == 'Shot 123, [0.900, 1.200] s, EQH'
    assert axes[0].texts[0].get_text() == (
        'Te fit quality\n$\\chi^2_{\\mathrm{red}}$=1.25\nwSSE=3.75')
    assert axes[1].texts[0].get_text() == (
        'Ne fit quality\n$\\chi^2_{\\mathrm{red}}$=2.5\nwSSE=7.5')
    assert axes[0].texts[0].get_position() == (0.02, 0.98)
    assert axes[1].texts[0].get_position() == (0.98, 0.02)
    plt.close(fig)


def test_spline_profile_plot_combines_te_ne_with_shared_title_and_metrics(monkeypatch):
    spline_data = pd.DataFrame({
        'R_mid_spl': [2.0, 2.3],
        'Te_spl': [150.0, 20.0],
        'ne_spl': [1.5e19, 2.0e18],
    })
    captured = {}
    fig, axes = plt.subplots(1, 2)

    def fake_ipfn_plots(**kwargs):
        captured.update(kwargs)
        return fig, axes

    monkeypatch.setattr(plotting, 'ipfn_plots', fake_ipfn_plots)

    fig, axes = plotting.plot_spline_profile_fit(
        x_core=[2.1],
        x_edge=[2.2],
        y_core={'Te': [100.0], 'Ne': [1.0e19]},
        y_edge={'Te': [80.0], 'Ne': [8.0e18]},
        y_err_core={'Te': [10.0], 'Ne': [1.0e18]},
        y_err_edge={'Te': [8.0], 'Ne': [8.0e17]},
        ds_out=spline_data,
        quantity=('Te', 'Ne'),
        x_labels=PROFILE_RADIUS_LABEL,
        y_labels=['Te [eV]', 'Ne [m$^{-3}$]'],
        knot_pos=[2.15],
        time_c=[0.9],
        time_e=[1.2],
        fit_metrics={
            'Te': {'reduced_chi2': 1.25, 'weighted_sse': 3.75},
            'Ne': {'reduced_chi2': 2.5, 'weighted_sse': 7.5},
        },
        fit_metrics_loc='center right',
        show=False,
        nshot=123,
        interval={'t_begin': 0.9, 't_end': 1.2},
        equilibrium='EQH',
    )

    assert captured['boxes'] == [0] * 6 + [1] * 6
    assert captured['n_boxes'] == 2
    assert captured['y_labels'] == ['Te [eV]', 'Ne [m$^{-3}$]']
    assert captured['title'] == 'Shot 123, [0.900, 1.200] s, EQH'
    assert captured['z_labels'] == 'Time [s]'
    assert captured['z_units'] == 's'
    assert captured['labels'] == [
        None, 'Core TS', None, 'Edge TS', 'Spline Fit', 'Knots',
        None, 'Core TS', None, 'Edge TS', 'Spline Fit', 'Knots',
    ]
    assert captured['y'][0].tolist() == [100.0]
    assert captured['y'][6].tolist() == [1.0e19]
    assert captured['z'][1].tolist() == [0.9]
    assert captured['z'][3].tolist() == [1.2]
    assert captured['plot_kw'][1]['vmin'] == 0.9
    assert captured['plot_kw'][1]['vmax'] == 1.2
    assert axes[0].texts[0].get_text() == (
        'Te fit quality\n$\\chi^2_{\\mathrm{red}}$=1.25\nwSSE=3.75')
    assert axes[1].texts[0].get_text() == (
        'Ne fit quality\n$\\chi^2_{\\mathrm{red}}$=2.5\nwSSE=7.5')
    assert axes[0].texts[0].get_position() == (0.98, 0.5)
    assert axes[1].texts[0].get_position() == (0.98, 0.5)
    plt.close(fig)


def test_profile_fit_plot_labels_time_and_uses_compact_title(monkeypatch):
    captured = {}

    def fake_ipfn_plots(**kwargs):
        captured.update(kwargs)
        fig, ax = plt.subplots()
        return fig, ax

    monkeypatch.setattr(plotting, 'ipfn_plots', fake_ipfn_plots)

    fig, ax = plot_profile_fit(
        x_core=[2.1],
        x_edge=[2.2],
        y_core=[100.0],
        y_edge=[80.0],
        y_err_core=[10.0],
        y_err_edge=[8.0],
        x_labels=PROFILE_RADIUS_LABEL,
        y_labels='Te [eV]',
        time_c=[0.9],
        time_e=[1.2],
        show_fit=False,
        show=False,
        nshot=123,
        interval={'t_begin': 0.9, 't_end': 1.2},
        equilibrium='EQH',
        x_core_out=[2.11],
        y_core_out=[0.0],
        y_err_core_out=[1.0],
        x_edge_out=[2.21],
        y_edge_out=[300.0],
        y_err_edge_out=[20.0],
    )

    assert captured['z_labels'] == 'Time'
    assert captured['z_units'] == 's'
    assert captured['title'] == 'Shot 123, [0.900, 1.200] s, EQH'
    assert captured['markersizes'] == [0, 16, 0, 16, 6, 6]
    assert [plot_kw['alpha'] for plot_kw in captured['plot_kw'][-2:]] == [
        0.12, 0.12]
    assert all('elinewidth' not in plot_kw for plot_kw in captured['plot_kw'][-2:])
    assert plotting._make_profile_title(
        'Custom title', 123, {'t_begin': 0.9, 't_end': 1.2}, 'EQH'
    ) == 'Custom title'
    plt.close(fig)