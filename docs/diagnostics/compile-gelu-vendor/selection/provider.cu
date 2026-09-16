// Generated PTX includes NVIDIA CUDA device math; see vendor/NVIDIA-CUDA-LICENSE.txt.
// Compile only offline with the pinned official libdevice input and local FTZ.
#include <math.h>
extern "C" __device__ float torch_rs_erf(float value) {
    return erff(value);
}
