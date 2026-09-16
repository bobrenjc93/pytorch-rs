# Generated NVIDIA math

The repository's original Rust, Python, CUDA and ABI-wrapper sources are MIT
licensed. The generated vendor math in `erf_provider.ptx` and `gelu.ptx` is
separate NVIDIA material, subject to the applicable CUDA distribution terms;
it is not relabeled MIT. `LicenseRef-NVIDIA-CUDA` in package metadata refers to
these included vendor terms:

- [CUDA 11.8 packaged LICENSE](../../docs/diagnostics/compile-gelu-vendor/notices/requested-CUDA11.8-LICENSE.txt)
- [CUDA 12.8 compiler LICENSE](../../docs/diagnostics/compile-gelu-vendor/notices/compiler-CUDA12.8-LICENSE.txt)
- [CUDA 13.0 LICENSE](../../docs/diagnostics/compile-gelu-linked/notices/CUDA13-LICENSE.txt)

`erf_provider.ll` is an original call/return wrapper. Offline CUDA 12.8 libNVVM
compiles it with the complete, unmodified official CUDA 11.8.89 libdevice
(SHA256 `1fc1bc8d4131d5a59a91fc26f4908886e231f842c9e83ae951979ff3df82bdb5`).
The resulting device function is PTX 8.7/sm75, SHA256
`d669b53d4f529a03c40d4b7eda3d0def5437e9a4edc52a130ed2d180075129de`.
It is linked only into compiled programs with live Erf.

`gelu.cu` is the separate eager kernel, generated offline with CUDA 13.0 nvcc
as PTX 9.0/sm75. It uses stock vendor `erff`; the original source does not
maintain coefficients or an approximation. Its image loads lazily and needs no
NVRTC. Common eager operations retain their existing PTX 6.0/sm50 image.

See the [generation and validation record](../../docs/diagnostics/compile-gelu-linked/README.md)
for reproducible commands and exact tool/input hashes. These are pinned,
maintainer-side generation tools, not production compiler dependencies. The
included licenses and measured checks do not certify broader deployment,
numerical equivalence, or legal compliance for other uses.
