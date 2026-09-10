# Matmul, unflatten and L1 diagnostic index

For the supported compiler scope, example and reproduction commands, use the
[compiled-matmul guide](../../compile-cuda-matmul.md). The
[integration history](../../composite-matmul-unflatten-l1-validation.md)
contains repair details, development checks and managed handoff notes.

## Current evidence

[Capture cbcbd84](postcommit-cbcbd84/README.md) measures the combined
implementation after both unflatten conversion repairs. Its fresh wheel passed
78 focused unflatten/L1 tests, CUDA compiler checks, an isolated compiled-program
proof and separate GPU 0,1 checks. The fixed math workload passed 18/18 trials,
the compiler corpus passed 38/38 cases, and all four fixed scoring shapes passed.
The separate matmul diagnostic passed 12/12 correctness cells with **79.42%
capped geometric parity**, retaining all six slower cells and every raw sample.

The native eager-backend graph is unfused. Its diagnostic is separate from the
fixed four-shape scoring workload and does not establish general Inductor parity.
The capture includes source/native/runtime identities, build and command
receipts, first-call/cache disclosures, both timing orders and integrity checks.

## Earlier committed captures

These records retain their original source/build identities and do not supply
current-candidate performance credit. Every listed diagnostic retained all 12
cells, including slower composed results. Each capture also passed all 18 fixed
math trials and the unchanged 38-case compiler corpus; full check results and
raw records remain at the links below.

| Capture | Measured scope | Separate matmul diagnostic |
| --- | --- | --- |
| [06a2496](postcommit-06a2496/README.md) | Composite after sizes-before-dim conversion repair; before native-size validation repair | 12/12 passed; 80.85% parity |
| [208e9bf](postcommit-208e9bf/README.md) | Initial composite with native unflatten dispatch; before both conversion repairs | 12/12 passed; 77.46% parity |
| [d8374ec](../compile-cuda-matmul/postcommit-d8374/README.md) | Compiled-matmul source PR only | 12/12 passed; 79.48% parity |

## Development records

The [source-PR development capture](../compile-cuda-matmul/development/README.md)
retains its uncommitted source manifest, checks, **76.45%** diagnostic result and
unsuccessful attempts. The [composite development record](../../composite-matmul-unflatten-l1-validation.md#development-validation-preserved)
retains its **78.87%** diagnostic result, broader test results and original local
log locations. Neither is clean-commit evidence. All original measurements,
failed attempts and provenance remain unchanged.
