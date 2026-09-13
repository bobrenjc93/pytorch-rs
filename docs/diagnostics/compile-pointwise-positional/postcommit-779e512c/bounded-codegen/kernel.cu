// torch_rs typed pointwise SSA v1; float32, no fast math
extern "C" __global__ void torch_rs_pointwise(
const float* x0, const float* x1, float* out, unsigned long long n) {
for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
i < n; i += (unsigned long long)blockDim.x * gridDim.x) {
const float v0 = x0[((i / 11ull) % 7ull) * 1ull];
const float v1 = x1[((i / 1ull) % 11ull) * 1ull];
const float v2 = (v0 < 0.0f ? 0.0f : v0);
const float v3 = (v1 < 0.0f ? 0.0f : v1);
const float v4 = __fsub_rn(v2, v3);
const float v5 = (v4 < 0.0f ? 0.0f : v4);
out[i] = v5;
}
}
