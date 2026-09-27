#!/usr/bin/env python
import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'aug_separatrix')))

from ts_profile_fitter.loading import load_shot_list, load_intervals
from ts_profile_fitter.pipeline import run_database
from get_connection_length_selectable import get_LcSOL


def parse_args():
    parser = argparse.ArgumentParser(description='Run TS profile fitter on shot database.')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file.')
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    def resolve(path, default):
        value = cfg.get(path, default)
        return value if os.path.isabs(value) else os.path.join(config_dir, value)

    shots = load_shot_list(resolve('shot_list', 'shot_list.csv'))
    database = load_intervals(resolve('database', cfg.get('intervals', 'intervals.csv')))
    output_dir = resolve('output_dir', 'results')
    fit_config = dict(cfg.get('fit_config') or {})
    for section in ('preprocess', 'fits', 'physics', 'plots', 'cache'):
        if section in cfg:
            fit_config[section] = cfg[section]
    if 'plots' in fit_config:
        fit_config['plots'] = dict(fit_config['plots'])
        for key in ('summary_directory', 'diagnostics_directory'):
            if key in fit_config['plots']:
                plot_directory = fit_config['plots'][key]
                fit_config['plots'][key] = (
                    plot_directory if os.path.isabs(plot_directory)
                    else os.path.join(config_dir, plot_directory)
                )
    if 'cache' in fit_config and 'directory' in fit_config['cache']:
        fit_config['cache'] = dict(fit_config['cache'])
        cache_directory = fit_config['cache']['directory']
        fit_config['cache']['directory'] = (
            cache_directory if os.path.isabs(cache_directory)
            else os.path.join(config_dir, cache_directory)
        )
    for key in ('diagnostic', 'equilibrium', 'fit_config_id'):
        if key in cfg:
            fit_config[key] = cfg[key]
    enriched = run_database(
        shots, database, resolve('data_dir', '.'), output_dir,
        fit_config=fit_config, connection_length_provider=get_LcSOL)
    enriched.to_pickle(resolve('output_database', 'enriched_database.pkl'))
    enriched.to_csv(resolve('output_database_csv', 'enriched_database.csv'), index=False)

if __name__ == '__main__':
    main()
