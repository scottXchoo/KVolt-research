"""Kneedle boundary tests; injected factories are explicit test doubles."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kvolt_runtime.config import GROUPS
from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy.knee import KneeAdapter
from kvolt_runtime.policy.types import Bracket, PolicyDependencyError, QualityPrediction


def quality(scores=(0.98, 0.97, 0.93, 0.80, 0.50), length=1024):
    return tuple(QualityPrediction(length, g, q, Bracket(length, length, 0))
                 for g, q in zip(GROUPS, scores))


def adapter_for(result):
    return KneeAdapter(lambda **kwargs: SimpleNamespace(all_knees=result))


def test_original_axis_raw_nonmonotonic_values_and_exact_configuration():
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(all_knees={16, 128, 8})
    curve = quality((0.97, 0.96, 0.98, 0.80, 0.50))
    result = KneeAdapter(factory).detect(tuple(reversed(curve)))
    assert calls == [dict(x=[8, 16, 32, 64, 128], y=[0.97, 0.96, 0.98, 0.80, 0.50],
                          S=1.0, curve='concave', direction='decreasing',
                          interp_method='interp1d', online=True)]
    assert (result.status, result.groups, result.reason) == ('found', (8, 16, 128), None)


def test_flat_is_exact_not_found_without_invoking_detector():
    def forbidden(**kwargs):
        pytest.fail('flat curve must bypass locator')
    result = KneeAdapter(forbidden).detect(quality((0.5,) * 5))
    assert (result.status, result.groups, result.reason) == ('not_found', (), 'flat_quality')


def test_one_binary64_step_is_not_flat():
    import math
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(all_knees=set())
    result = KneeAdapter(factory).detect(quality((0.5, 0.5, 0.5, 0.5, math.nextafter(0.5, 1))))
    assert len(calls) == 1
    assert result.reason == 'no_knee'


@pytest.mark.parametrize('result', [set(), frozenset(), [], ()])
def test_empty_detection_is_normal(result):
    found = adapter_for(result).detect(quality())
    assert (found.status, found.groups, found.reason) == ('not_found', (), 'no_knee')


@pytest.mark.parametrize('result', [{8}, {128}, {8.0, 32.0, 128.0}, frozenset({16, 64}),
                                   [128, 16, 128, 8], (64.0, 8.0, 64.0)])
def test_endpoints_and_multiple_original_candidates_are_valid(result):
    found = adapter_for(result).detect(quality())
    assert found.status == 'found'
    assert found.groups == tuple(sorted(set(int(g) for g in result)))


@pytest.mark.parametrize('result', [None, 32, '32', {True}, {'32'}, {'knee': 32},
                                   iter([32]), {float('nan')}, {float('inf')},
                                   {32, 63}, {16.1}, [32, True], (8, float('nan'))])
def test_invalid_return_is_recovered_in_full(result):
    found = adapter_for(result).detect(quality())
    assert (found.status, found.groups) == ('error_recovered', ())
    assert found.reason == 'detector_error'
    assert found.error_type and found.error_message


@pytest.mark.parametrize('where', ['construction', 'property'])
def test_exception_is_recorded_for_same_request_recovery(where):
    class Broken:
        @property
        def all_knees(self):
            raise ArithmeticError('locator failed')
    def factory(**kwargs):
        if where == 'construction':
            raise ArithmeticError('locator failed')
        return Broken()
    found = KneeAdapter(factory).detect(quality())
    assert (found.status, found.error_type, found.error_message) == (
        'error_recovered', 'ArithmeticError', 'locator failed')


@pytest.mark.parametrize('error', [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(error):
    def factory(**kwargs):
        raise error()
    with pytest.raises(error):
        KneeAdapter(factory).detect(quality())


@pytest.mark.parametrize('bad', [lambda q: q[:-1], lambda q: q + (q[0],),
                                 lambda q: list(q), lambda q: (q[0],) * 5,
                                 lambda q: (replace(q[0], length=1536, bracket=Bracket(1536, 1536, 0)),) + q[1:]])
def test_invalid_quality_is_fatal_before_detector(bad):
    calls = []
    with pytest.raises(ContractError):
        KneeAdapter(lambda **kwargs: calls.append(kwargs)).detect(bad(quality()))
    assert calls == []


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -0.1, 1.1, True, '0.5'])
def test_corrupt_raw_quality_cannot_become_detector_recovery(score):
    rows = quality()
    object.__setattr__(rows[0], 'score', score)
    with pytest.raises(ContractError):
        adapter_for(set()).detect(rows)


@pytest.mark.parametrize('package,version', [('kneed', '0.8.5'), ('scipy', '1.15.2')])
def test_wrong_dependency_version_fails_at_startup(monkeypatch, package, version):
    import kvolt_runtime.policy.knee as module
    def fake_import(name):
        return SimpleNamespace(__version__=version if name == package else
                               {'kneed': '0.9.0', 'scipy': '1.15.3'}[name],
                               KneeLocator=lambda **kwargs: None)
    monkeypatch.setattr(module.importlib, 'import_module', fake_import)
    with pytest.raises(PolicyDependencyError, match=package):
        KneeAdapter()


def test_missing_dependency_is_startup_error(monkeypatch):
    import kvolt_runtime.policy.knee as module
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(module.importlib, 'import_module', missing)
    with pytest.raises(PolicyDependencyError):
        KneeAdapter()


def test_actual_pinned_kneedle_differential():
    """Required environment check: missing/wrong pins fail, never silently skip."""
    adapter = KneeAdapter()
    from kneed import KneeLocator
    import numpy as np
    curves = [(0.98, 0.97, 0.93, 0.80, 0.50), (0.97, 0.96, 0.98, 0.80, 0.50),
              (1.0, 0.9, 0.7, 0.4, 0.0), (0.8, 0.79, 0.78, 0.77, 0.76)]
    for scores in curves:
        direct = KneeLocator(x=list(GROUPS), y=list(scores), S=1.0, curve='concave',
                             direction='decreasing', interp_method='interp1d', online=True)
        expected = tuple(sorted(int(g) for g in direct.all_knees))
        observed = adapter.detect(quality(scores))
        assert observed.groups == expected
        assert observed.status == ('found' if expected else 'not_found')
    assert adapter_for({np.int64(8), np.float64(128)}).detect(quality()).groups == (8, 128)
    assert adapter_for([np.int64(128), np.float64(8), np.int64(128)]).detect(quality()).groups == (8, 128)
    assert adapter_for(np.array([8, 128])).detect(quality()).status == 'error_recovered'


def test_actual_published_quality_curves_differential():
    """Compare all 15 published knots and 14 interiors against official Kneedle."""
    import os
    from pathlib import Path
    from kvolt_runtime.policy.lut import MeasuredLUTRepository
    from kvolt_runtime.policy.types import QUALITY_LENGTHS

    adapter = KneeAdapter()  # Check pins even if this optional data source is absent.
    from kneed import KneeLocator
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    bundle = (Path(configured) if configured else Path(__file__).resolve().parents[2] /
              'KVolt-lut-source/data/luts/agx2-repeat5-20261103')
    if not bundle.is_dir() and configured is None:
        pytest.skip('sealed published bundle unavailable: set KVOLT_LUT_BUNDLE')
    repo = MeasuredLUTRepository(bundle)  # Verifies source pins and all file hashes.
    lengths = list(QUALITY_LENGTHS) + [n + 256 for n in QUALITY_LENGTHS[:-1]]
    for length in lengths:
        rows = repo.quality(length)
        raw = [row.score for row in rows]
        expected = () if len(set(raw)) == 1 else tuple(sorted(int(g) for g in KneeLocator(
            x=list(GROUPS), y=raw, S=1.0, curve='concave', direction='decreasing',
            interp_method='interp1d', online=True).all_knees))
        observed = adapter.detect(rows)
        assert observed.groups == expected, length
        assert observed.status == ('found' if expected else 'not_found'), length
