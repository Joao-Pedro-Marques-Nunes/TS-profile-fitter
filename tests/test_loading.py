import numpy as np
import pandas as pd
from types import SimpleNamespace

import ts_profile_fitter.loading as loading
from ts_profile_fitter.loading import _build_vta_dataframe


class DummyObject:
    def __init__(self, data):
        self.data = data


class DummyVTA:
    def __init__(self):
        self.status = True

    def __call__(self, name):
        if name == 'TIM_CORE':
            return DummyObject(np.array([0.1, 0.2]))
        if name == 'TIM_EDGE':
            return DummyObject(np.array([0.3, 0.4]))
        raise KeyError(name)

    def getobject(self, name):
        if name == 'R_core':
            return DummyObject(np.array([1.01, 1.02]))
        if name == 'Z_core':
            return DummyObject(np.array([0.0, 0.1, 0.2]))
        if name == 'R_edge':
            return DummyObject(np.array([1.03, 1.04]))
        if name == 'Z_edge':
            return DummyObject(np.array([0.3, 0.4, 0.5]))
        if name == 'Te_c':
            return DummyObject(np.array([[2.0, 2.1, 2.2], [2.3, 2.4, 2.5]]))
        if name == 'SigTe_c':
            return DummyObject(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
        if name == 'Ne_c':
            return DummyObject(np.array([[10.0, 10.1, 10.2], [10.3, 10.4, 10.5]]))
        if name == 'SigNe_c':
            return DummyObject(np.array([[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]))
        if name == 'Te_e':
            return DummyObject(np.array([[3.0, 3.1, 3.2], [3.3, 3.4, 3.5]]))
        if name == 'SigTe_e':
            return DummyObject(np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]))
        if name == 'Ne_e':
            return DummyObject(np.array([[20.0, 20.1, 20.2], [20.3, 20.4, 20.5]]))
        if name == 'SigNe_e':
            return DummyObject(np.array([[0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]))
        raise KeyError(name)


def test_build_vta_dataframe_handles_time_and_channel_vectors():
    df = _build_vta_dataframe(DummyVTA())

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert set(['system', 'time', 'r', 'z', 'Te', 'Te_err', 'Ne', 'Ne_err']).issubset(df.columns)
    assert np.allclose(df.loc[df['system'] == 'core', 'time'].unique(), [0.1, 0.2])
    assert np.allclose(df.loc[df['system'] == 'core', 'z'].unique(), [0.0, 0.1, 0.2])


def test_fast_wmhd_dwdt_uses_selected_gqh_and_converts_mw(monkeypatch):
    calls = {}

    class DummyGQH:
        status = True

        def getobject(self, name, tbeg, tend):
            assert (name, tbeg, tend) == ('Wmhd', 1.0, 1.2)
            return np.array([0.0, 1.0, 2.0])

        def gettimebase(self, name, tbeg, tend):
            assert (name, tbeg, tend) == ('Wmhd', 1.0, 1.2)
            return np.array([1.0, 1.1, 1.2])

    def open_shotfile(shot_id, diagnostic, **kwargs):
        calls['shotfile'] = (shot_id, diagnostic, kwargs)
        return DummyGQH()

    def dwdt_field(time, wmhd, elm_times, shot_id, phase_begin, phase_end):
        calls['field'] = (time, wmhd, elm_times, shot_id, phase_begin, phase_end)
        return 2.5, 0.25

    monkeypatch.setattr(loading, '_open_optional_shotfile', open_shotfile)
    monkeypatch.setitem(
        __import__('sys').modules,
        'get_functions',
        SimpleNamespace(dWdt_Field=dwdt_field),
    )

    dwdt, dwdt_err = loading._load_fast_wmhd_dwdt(
        12345,
        1.0,
        1.2,
        SimpleNamespace(exp='EXP', diag='GQH', ed=7),
        np.array([1.05, 1.15]),
        (0.6, 0.8),
    )

    assert calls['shotfile'] == (12345, 'GQH', {'exp': 'EXP', 'diag': 'GQH', 'ed': 7})
    assert calls['field'][3:] == (12345, 0.6, 0.8)
    assert np.isclose(dwdt, 2.5e6)
    assert np.isclose(dwdt_err, 0.25e6)
