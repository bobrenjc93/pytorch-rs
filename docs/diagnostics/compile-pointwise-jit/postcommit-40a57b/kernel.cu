// torch_rs typed pointwise SSA v1; float32, no fast math
extern "C" __global__ void torch_rs_pointwise(
const float* x0, const float* x1, float* out, unsigned long long n) {
for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
i < n; i += (unsigned long long)blockDim.x * gridDim.x) {
const float v0 = x0[i];
const float v1 = x1[i];
const float v3 = __uint_as_float(0x3f36add6u);
const float v4 = __fmul_rn(v0, v3);
const float v5 = sinf(v4);
const float v6 = __fmul_rn(v5, v5);
const float v7 = cosf(v1);
const float v8 = fmaf(v5, v5, -v7);
const float v10 = __uint_as_float(0x3d01f751u);
const float v11 = __fmul_rn(v5, v10);
const float v12 = fmaf(v5, v10, v8);
const float v13 = (v12 < 0.0f ? 0.0f : v12);
out[i] = v13;
}
}
