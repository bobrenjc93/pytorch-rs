def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return False


from .distributed_c10d import (
    get_default_backend_for_device as get_default_backend_for_device,
)
from .distributed_c10d import get_pg_count as get_pg_count
from .distributed_c10d import get_node_local_rank as get_node_local_rank
from .distributed_c10d import is_gloo_available as is_gloo_available
from .distributed_c10d import is_initialized as is_initialized
from .distributed_c10d import is_mpi_available as is_mpi_available
from .distributed_c10d import is_nccl_available as is_nccl_available
from .distributed_c10d import is_ucc_available as is_ucc_available
from .distributed_c10d import is_xccl_available as is_xccl_available
