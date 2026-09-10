# Local integration validation

These are worktree-local checks, not a claim that committed final-source evidence, official prompt evaluations, or remote CI passed. The native wheel was built at clean integrated HEAD before diagnostic/documentation repairs. Production sources did not change afterward.

## Test outcomes

- `focused`: Ran 159 tests in 55.427s; OK (skipped=6)
- `diagnostic-tests`: Ran 5 tests in 15.292s; OK
- `dev-only-tests`: Ran 5 tests in 0.000s; OK (skipped=4)
- `two-device`: Ran 6 tests in 8.076s; OK
- `python-full`: Ran 5331 tests in 499.369s; OK (skipped=11)
- `managed312-known-failures`: Ran 13 tests in 0.034s; FAILED (failures=2)
- `baseline-managed312-known-failures`: Ran 13 tests in 0.049s; FAILED (failures=2)
- `factory-candidate-cache-test-seed6`: Ran 5 tests in 0.002s; FAILED (failures=3)
- `factory-baseline-cache-test-seed6`: Ran 5 tests in 0.003s; FAILED (failures=3)
- `python314-full`: Ran 5331 tests in 532.069s; OK (skipped=11)
- `rust-cuda-add`: test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.48s
- `rust-cuda`: test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 153 filtered out; finished in 0.46s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 85 filtered out; finished in 0.00s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.00s; test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.45s; test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 102 filtered out; finished in 0.00s
- `rust`: 355 passed.
- `rust-bindings-managed`: 366 passed.

## Unchanged evaluator outcomes

- Compile: torch.compile coverage parity is 100.0/100 from 38/38 reference-eligible full corpus cases. Covered categories: tensor_arithmetic, broadcasting, modules_parameters_buffers, inference, training_autograd, python_control_flow, graph_breaks_fullgraph, dynamic_shapes_symbolics, mutation_aliasing_views, containers_pytrees, decompositions, custom_functions, recompilation_guards, dtype_device_transitions; all other weighted categories remain zero-credit until they have passing reference-compilable torch_rs cases.
- CUDA math: 2/6; all unsupported cases remain zero credit.
- `cuda-performance-fresh`: 4/4 correct; capped score 100.000000%; geometric mean speed ratio 1.194364.
- `cuda-performance-warm`: 4/4 correct; capped score 92.103646%; geometric mean speed ratio 1.021424.
- `cuda-performance-confirm-1`: 4/4 correct; capped score 100.000000%; geometric mean speed ratio 1.151067.
- `cuda-performance-confirm-2`: 4/4 correct; capped score 79.301901%; geometric mean speed ratio 0.728105.

Raw logs, environment/build receipts, full reports and exact command/exit receipts are in this directory. Setup failures and all failed tests are retained. `independent-review.md` covers all ten rubrics qualitatively; official numerical prompt scores were not available through the exposed tools. Historical raw evidence and all frozen evaluators were preserved.
