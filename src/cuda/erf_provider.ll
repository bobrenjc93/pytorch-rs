; Original ABI wrapper; math is provided only by the separately supplied library.
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare float @__nv_erff(float)

define float @torch_rs_erf(float %value) {
entry:
  %result = call float @__nv_erff(float %value)
  ret float %result
}

!nvvmir.version = !{!0}
!0 = !{i32 2, i32 0, i32 3, i32 2}
