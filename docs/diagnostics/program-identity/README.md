# Program-identity public diagnostic

This is the non-scoring consumer of the [frozen v2 protocol](protocol.md), whose
SHA256 is `d6e8f51d024b451f8afed33d743647d46396425724b918a599f7b32274f95d5b`.
It does not produce a qualification verdict or change the fixed evaluation.
Historical A measurements do not qualify this implementation.

[Development validation](validation.md) records the implementation checks and
preserved failures. The [clean post-commit attempt](postcommit-47b97f2/README.md)
stopped at the first reference leg's runtime-provenance check; the full public
comparison remains incomplete.

`consumer.py` uses only the standard library until a worker imports its selected
framework. Its `build`, `check`, `freeze`, `preflight`, `leg`, and `verify` subcommands create
new attempt directories and never overwrite records. An exception retains a
failed record. Use `.json.gz` output names for deterministic compressed raw
records. Keep attempt directories beneath an ignored worktree-local directory
until canonical evidence delivery; source checks require a clean checkout.

Run the hardware-free controls before capture:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/diagnostics/program-identity/consumer.py check \
  --output target/program-identity/check-1/result.json.gz
```

The check includes complete synthetic eight-leg acceptance and missing/failing
leg, missing sample/cell, changed binding/runtime, dirty source, lifetime,
ownership, signed-zero, finite reference tolerance, selected-invocation,
fixed-cap and stable-hint negative controls. Synthetic records are test data,
never hardware evidence. `freeze --controls <check-result> --output <fresh-path>`
requires the current implementation/tooling to have a clean non-baseline commit.
Burner's canonical lifecycle owns that commit; the implementation agent must
not create one manually.

For each separately owned clean B/C checkout, use its worktree-local build
interpreter and `build --source-root <checkout> --commit <full-sha> --output
<fresh-path>`. The command records release/locked Maturin invocation and checks
wheel Python bytes against that checkout. Install each resulting wheel into its
own worktree-local environment through the canonical evidence phase. The worker
checks every installed Python file and native extension against the wheel and
source again. Do not use an editable install or import reference PyTorch in a
native worker. Reference environments must contain locked `2.13.0+cu130`.

Under the canonical `cpu-heavy`/`gpu` lease, first run a separate candidate
`preflight --build-record <C-build> --freeze <freeze> --runtime <runtime> --nvrtc <compiler> --output
<fresh-path>` process with `CUDA_VISIBLE_DEVICES=0`. This untimed structural phase
checks the fixed over-cap fixture before any dependent ordinary measurement.
Pass its successful record to every leg with `--preflight`; retain a failed
preflight and stop that attempt without changing the workload.

Then launch eight **serial fresh
processes** with `CUDA_VISIBLE_DEVICES=0`, each running:

```sh
<worktree-local-python> docs/diagnostics/program-identity/consumer.py leg \
  --index <0-through-7> --build-record <paired-build-result> \
  --freeze <freeze-result> --preflight <preflight-result> \
  --runtime <same-absolute-libcudart-path> --nvrtc <same-absolute-libnvrtc-path> \
  --output <fresh-leg-directory>/result.json.gz
```

The fixed order is B-native, B-reference, C-native, C-reference, C-reference,
C-native, B-reference, B-native. The command fixes implementation from index;
there is no backend/timing option. All legs use the identical explicitly pinned
CUDA runtime barrier and verify GPU0's H100 UUID in inventory and driver. Runtime identity comes from
`cudaDeviceGetPCIBusId`, matched uniquely against the independent physical
inventory; libcudart has no standalone UUID getter. Driver, runtime and reference
must each see exactly one device at logical CUDA index zero.
Both library pins must resolve to nonempty files inside this worktree. Before
framework import, the worker sets `TORCH_RS_CUDART` and `TORCH_RS_NVRTC` to those
exact paths. Native workers retain the selected NVRTC handle outside timing;
first-call timings therefore exclude its initial dynamic-library load equally
for B and C. `dladdr` binds actual runtime/compiler symbol providers to the pins.
Post-run mapping checks reject additional CUDA runtimes and missing/different
NVRTC providers; native kernel getters verify the existing precise options and
compiler version outside timing. The offline verifier repeats these checks and
requires identical native compiler bytes/options across B/C.

Paths, hashes, loaded CUDA/compiler libraries, pre/post inventory and complete
raw observations are retained. Fresh processes/directories do not clear hardware
caches. Preserve foreign jobs. Each candidate worker executes separately labeled
receipt controls after ordinary timing. An over-cap fixture failure ends that
attempt; never adjust `range(320)` or substitute a workload.

`verify <eight-ordered-records> --output <fresh-path>` is hardware-free. It retains
an inventory of compressed raw-record hashes, all failures, cold observations,
dispersion and 72 per-cell/per-order ratios. Native B/C uses exact finite bits;
reference comparison uses finite `1e-4 + 1e-4 * abs(reference)` and exact metadata.
No outlier filtering, pooling or parity score is produced. A later canonical
evidence-only delivery should retain these immutable files and a concise result
manifest here. No clean hardware results are claimed by the tooling commit.
