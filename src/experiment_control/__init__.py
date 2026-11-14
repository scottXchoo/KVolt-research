"""Detachable, stdlib-only experiment environment control and observation."""
from .contracts import (Control, EnvironmentError, MetricRule, Probe, Reading,
                        ReadinessError, RestorationError, ViolationError)
from .guard import Environment, controlled
from .lock import FileLock


def environment_from_profile(*args, **kwargs):
    """Lazily load Linux adapters; generic API imports have no platform probes."""
    from .profile import environment_from_profile as build
    return build(*args, **kwargs)

__all__ = ['Environment', 'controlled', 'MetricRule', 'Reading', 'Control', 'Probe',
           'FileLock', 'EnvironmentError', 'ReadinessError', 'RestorationError', 'ViolationError',
           'environment_from_profile']
