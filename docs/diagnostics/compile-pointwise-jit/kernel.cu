// torch_rs typed pointwise SSA v1; float32, no fast math
extern "C" __global__ void torch_rs_pointwise(
const float* x0, const float* x1, float* out, unsigned long long n) {
for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
i < n; i += (unsigned long long)blockDim.x * gridDim.x) {
const float v0 = x0[i];
const float v1 = x1[i];
const float v2 = __uint_as_float(0x3f36add6u);
const float v3 = (v0 * v2);
const float v4 = sinf(v3);
const float v5 = (v4 * v4);
const float v6 = cosf(v1);
const float v7 = fmaf(v4, v4, -v6);
const float v8 = __uint_as_float(0x3d01f751u);
const float v9 = (v4 * v8);
const float v10 = fmaf(v4, v8, v7);
const float v11 = (v10 < 0.0f ? 0.0f : v10);
out[i] = v11;
}
}
