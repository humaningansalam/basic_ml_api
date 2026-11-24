from datetime import UTC, datetime, timedelta

MODEL_RETENTION_PERIOD = timedelta(weeks=1)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(UTC)


def stale_model_cutoff() -> datetime:
    """Return the cutoff used to decide whether a model is stale."""
    return utc_now() - MODEL_RETENTION_PERIOD
