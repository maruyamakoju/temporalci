"""Unified exception hierarchy for TemporalCI.

All domain-specific errors inherit from :class:`TemporalCIError` so callers
can catch the whole family with a single ``except`` clause when appropriate.
"""

from __future__ import annotations


class TemporalCIError(Exception):
    """Base exception for all TemporalCI errors."""


class ConfigError(TemporalCIError, ValueError):
    """Invalid suite configuration, YAML schema, or prompt source error.

    Inherits from ``ValueError`` so existing ``except ValueError`` handlers
    continue to work during the migration period.
    """


class AdapterError(TemporalCIError, RuntimeError):
    """Failure within a model adapter (loading, inference, encoding)."""


class MetricError(TemporalCIError, RuntimeError):
    """Failure within a metric evaluation backend."""


class CoordinatorError(TemporalCIError, RuntimeError):
    """Failure in the distributed coordinator or worker."""
