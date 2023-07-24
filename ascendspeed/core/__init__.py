import ascendspeed.core.parallel_state
import ascendspeed.core.tensor_parallel
import ascendspeed.core.utils

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
]