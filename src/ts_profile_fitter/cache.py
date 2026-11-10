"""Small, versioned disk caches for expensive physics inputs."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path


class PhysicsCache:
    """Cache plain Python physics inputs and connection lengths by key."""

    def __init__(self, directory, enabled=None, reuse_failed=False,
                 provider_id='get_LcSOL_selectable_v1', mapped_data_enabled=None,
                 physics_inputs_enabled=None, connection_length_enabled=None,
                 on_new_cache=None):
        self.directory = Path(directory)
        default_enabled = bool(enabled) if enabled is not None else False
        self.enabled_by_kind = {
            'mapped_data': default_enabled if mapped_data_enabled is None
            else bool(mapped_data_enabled),
            'physics': default_enabled if physics_inputs_enabled is None
            else bool(physics_inputs_enabled),
            'connection_length': default_enabled if connection_length_enabled is None
            else bool(connection_length_enabled),
        }
        self.reuse_failed = bool(reuse_failed)
        self.provider_id = str(provider_id)
        self.on_new_cache = on_new_cache
        if any(self.enabled_by_kind.values()):
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, kind, key_data):
        payload = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]
        return self.directory / f'{kind}_{digest}.pkl'

    def _read(self, path):
        try:
            with path.open('rb') as stream:
                value = pickle.load(stream)
        except (OSError, EOFError, pickle.PickleError, ValueError):
            return None
        if value.get('status') == 'ok' or self.reuse_failed:
            return value
        return None

    def _write(self, path, value):
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f'.{path.name}.', dir=str(path.parent))
        try:
            with os.fdopen(fd, 'wb') as stream:
                pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get_or_compute(self, kind, key_data, compute):
        if not self.enabled_by_kind.get(kind, False):
            value = compute()
            if self.on_new_cache is not None:
                self.on_new_cache(kind, key_data, value)
            return value, False
        path = self._path(kind, key_data)
        if path.exists():
            cached = self._read(path)
            if cached is not None:
                if cached['status'] == 'ok':
                    return cached['value'], True
                raise RuntimeError(cached['error'])
        try:
            value = compute()
        except Exception as exc:
            self._write(path, {'status': 'failed', 'error': str(exc)})
            raise
        self._write(path, {'status': 'ok', 'value': value})
        if self.on_new_cache is not None:
            self.on_new_cache(kind, key_data, value)
        return value, False

    def is_enabled(self, kind):
        return self.enabled_by_kind.get(kind, False)

    def physics_inputs(self, shot_id, interval, equilibrium, compute):
        key = {
            'version': 'physics_inputs_v2',
            'shot_id': int(shot_id),
            't_begin': float(interval['t_begin']),
            't_end': float(interval['t_end']),
            'equilibrium': equilibrium,
        }
        return self.get_or_compute('physics', key, compute)

    def connection_length(self, shot_id, interval, dR, compute, equilibrium='EQH'):
        key = {
            'version': 'connection_length_v1',
            'provider': self.provider_id,
            'equilibrium': equilibrium,
            'shot_id': int(shot_id),
            'time': 0.5 * (float(interval['t_begin']) + float(interval['t_end'])),
            'dR': float(dR),
        }
        return self.get_or_compute('connection_length', key, compute)

    def mapped_data(self, shot_id, diagnostic, equilibrium, mapping_method,
                    r_edge_offset, compute):
        key = {
            'version': 'mapped_data_v1',
            'shot_id': int(shot_id),
            'diagnostic': diagnostic,
            'equilibrium': equilibrium,
            'mapping_method': mapping_method,
            'r_edge_offset': float(r_edge_offset),
        }
        return self.get_or_compute('mapped_data', key, compute)
