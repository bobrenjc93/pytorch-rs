from datetime import timedelta

from ._constants import _DEFAULT_PG_TIMEOUT


__all__ = ["default_pg_timeout"]

# Default process group wide timeout, if applicable.
# This only applies to the non-nccl backends.
default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT
