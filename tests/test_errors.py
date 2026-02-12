from __future__ import annotations

import pytest

from temporalci.errors import (
    AdapterError,
    ConfigError,
    CoordinatorError,
    MetricError,
    TemporalCIError,
)


def test_all_errors_inherit_from_base() -> None:
    for cls in (ConfigError, AdapterError, MetricError, CoordinatorError):
        assert issubclass(cls, TemporalCIError)


def test_config_error_is_valueerror() -> None:
    assert issubclass(ConfigError, ValueError)
    with pytest.raises(ValueError):
        raise ConfigError("bad config")


def test_adapter_error_is_runtimeerror() -> None:
    assert issubclass(AdapterError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise AdapterError("adapter broke")


def test_metric_error_is_runtimeerror() -> None:
    assert issubclass(MetricError, RuntimeError)


def test_coordinator_error_is_runtimeerror() -> None:
    assert issubclass(CoordinatorError, RuntimeError)


def test_catch_all_domain_errors() -> None:
    """Verify a single except clause can catch all domain errors."""
    for cls in (ConfigError, AdapterError, MetricError, CoordinatorError):
        with pytest.raises(TemporalCIError):
            raise cls("test")


def test_backward_compat_alias_suite_validation_error() -> None:
    from temporalci.config import SuiteValidationError

    assert SuiteValidationError is ConfigError


def test_backward_compat_alias_prompt_source_error() -> None:
    from temporalci.prompt_sources import PromptSourceError

    assert PromptSourceError is ConfigError
