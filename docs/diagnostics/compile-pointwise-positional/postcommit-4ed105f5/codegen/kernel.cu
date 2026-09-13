// torch_rs typed pointwise SSA v1; float32, no fast math
extern "C" __global__ void torch_rs_pointwise(
const float* x0, const float* x1, float* out, unsigned long long n, float s0) {
for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
i < n; i += (unsigned long long)blockDim.x * gridDim.x) {
const float v0 = x0[i];
const float v2 = s0;
const float v3 = __fmul_rn(v0, v2);
const float v6 = (v3 < 0.0f ? 0.0f : v3);
out[i] = v6;
}
}
