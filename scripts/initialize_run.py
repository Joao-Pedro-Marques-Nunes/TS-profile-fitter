#!/usr/bin/env python
"""Create a run-specific TS profile fitter configuration."""

import argparse
import copy
import os
from pathlib import Path

import yaml


def initialize_config(config_path, output_dir, config_output=None, in_place=False):
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    if in_place:
        destination = config_path
    elif config_output is not None:
        destination = Path(config_output)
        if not destination.is_absolute():
            destination = (Path.cwd() / destination).resolve()
    else:
        destination = config_path.with_name(f'{config_path.stem}_run.yaml')

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    with config_path.open('r') as stream:
        config = yaml.safe_load(stream) or {}
    config = copy.deepcopy(config)

    def path_value(path):
        path = Path(path)
        if path.is_absolute():
            return str(path)
        if in_place:
            return os.path.relpath((config_dir / path).resolve(), config_dir)
        return str((config_dir / path).resolve())

    for key in ('data_dir', 'shot_list', 'database', 'intervals'):
        if key in config:
            config[key] = path_value(config[key])

    database_name = Path(config.get('database', 'database.pkl')).stem
    config['output_dir'] = path_value(output_dir)
    config['output_database'] = path_value(
        output_dir / f'{database_name}_fitted.pkl')
    config['output_database_csv'] = path_value(
        output_dir / f'{database_name}_fitted.csv')

    cache = config.setdefault('cache', {})
    cache['directory'] = path_value(output_dir / 'cache')

    plots = config.setdefault('fit_config', {}).setdefault('plots', {})
    plots['summary_directory'] = path_value(output_dir / 'profile_summary')
    plots['diagnostics_directory'] = path_value(
        output_dir / 'profile_diagnostics')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False, default_flow_style=False)
    return destination


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a run-specific TS profile fitter YAML configuration.')
    parser.add_argument('--config', required=True, help='Template YAML configuration.')
    parser.add_argument(
        '--output-dir', required=True,
        help='Root directory for this run and all derived outputs.')
    parser.add_argument(
        '--config-output',
        help='Destination for the generated YAML (default: <config>_run.yaml).')
    parser.add_argument(
        '--in-place', action='store_true',
        help='Rewrite the input YAML instead of creating a generated copy.')
    return parser.parse_args()


def main():
    args = parse_args()
    destination = initialize_config(
        args.config, args.output_dir, args.config_output, args.in_place)
    print(f'Initialized run configuration: {destination}')


if __name__ == '__main__':
    main()