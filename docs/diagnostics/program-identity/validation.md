# Program-identity development validation — 2026-09-16

The implementation reuses exact executable identities in the existing executor
LRU, including equivalent Programs after preparation eviction. Checked host
preparation selects one direct or VM executable; preparation hits retain the
existing module, plan and accounting. Direct emission uses the frozen inclusive
256-instruction, 128-register and 65,536-complete-source-byte limits.

These are development checks against uncommitted changes from accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b`. The clean eight-leg public
B/C/reference protocol and full qualification **have not run**. Burner must
first commit implementation/tooling and freeze the consumer from clean source;
this agent did not create a commit. No public speedup, parity or score is claimed.

| Final check | Result |
| --- | --- |
| Rust library, Python bindings enabled, GPU0, serial tests | 290 passed |
| Complete Python pointwise suite on H100 | 382 run, 9 skipped, no failures |
| Real native bridge and injected compiler/module failures | 9 passed |
| Hardware-free public consumer tests | 33 passed |
| Consumer acceptance/rejection controls | Passed |
| Full fixed candidate structural controls, untimed | Passed |
| Clippy library/tests with Python bindings and `-D warnings` | Passed |
| Rust formatting | Passed |

Coverage includes exact cap boundaries, original-graph/dead-input admission,
numerical hint ordering, distinct Programs and address maps, preparation
eviction/reuse, direct/VM domains, output and owner lifetimes, capacity budgets,
reset/pruning, and failures before cache publication. The frozen consumer's
over-cap fixture produced 325 instructions and selected VM for both sizes;
the other five families selected direct code. These candidate-only controls
are development evidence, not canonical preflight or public timing.

GPU tests used only logical GPU0, H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. Worktree-local
libraries pinned CUDA runtime 13.0 and NVRTC 13.0; installed nvcc 12.6 was not
the production JIT compiler. The environment used Python 3.12.12, Rust/Cargo
1.92.0 and PyTorch 2.13.0+cu130. The release wheel's Python and extension bytes
were checked against the installed development package. No additional GPU was
reserved, so this evidence makes no multi-device claim.

The [manifest](validation-manifest.json) binds source/test hashes and all 61
records in the [compressed raw archive](development-validation-20260916.tar.gz).
It includes build/test logs, runtime/compiler hashes, GPU snapshots, wheel
verification, controls, and earlier failed attempts. Earlier failures exposed
bridge error ordering, private export leakage, outdated cache/legacy-API test
assumptions, test-runner/interpreter setup, an invalid new Rust fixture, the
driver's alternate module-rejection code, and a nonexistent runtime UUID getter.
Those failures remain recorded alongside their passing corrections. The frozen
public workloads and historical diagnostic scripts were unchanged.

See the [consumer instructions](README.md) and [frozen protocol](protocol.md)
for the remaining canonical capture phase. The verifier requires every ordered
leg and raw sample and rejects incomplete or failed attempts.
