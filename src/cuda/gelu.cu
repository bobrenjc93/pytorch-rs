// Native eager default GELU. Generate PTX offline; normal builds need no compiler.
// NVIDIA device math is subject to the vendor terms recorded with this image.
#include <math.h>
extern "C" __global__ void gelu_f32(const float* input, float* output,
                                    unsigned long long count) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long)blockDim.x * gridDim.x) {
        float x = input[i];
        output[i] = (x * 0.5f) * (1.0f + erff(x * 0.70710678118654752440f));
    }
}
