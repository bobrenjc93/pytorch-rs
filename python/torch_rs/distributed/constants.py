from datetime import timedelta

from ._constants import _DEFAULT_PG_TIMEOUT


__all__ = ["default_pg_timeout"]

# Default process-group-wide timeout for non-NCCL backends.
default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT
