"""Hardware-free controls for the non-scoring protocol; synthetic data are not evidence."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

PATH = Path(__file__).resolve().parents[1] / "docs/diagnostics/program-identity/consumer.py"
SPEC = importlib.util.spec_from_file_location("program_identity_consumer", PATH)
consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consumer)


def tensor(value="0x1.0000000000000p+0"):
    return {"kind": "tensor", "shape": [1], "stride": [1], "dtype": "torch.float32",
            "device": "cuda:0", "requiresGrad": False, "alias": [], "values": [value]}


def selection(kind="direct", instructions=3, identity="01"):
    return {"identity": identity, "kind": kind, "instructions": instructions, "registers": 2,
            "source": "source", "sourceSha256": consumer.sha(b"source"),
            "ptx": "ptx", "ptxSha256": consumer.sha(b"ptx"),
            "options": list(consumer.COMPILER_OPTIONS), "nvrtcVersion": [13, 0]}


def reports():
    """Minimal complete schema fixture; never emitted as a hardware capture."""
    records = []
    for index, (label, implementation) in enumerate(consumer.ORDER):
        cells = []
        for name, _ in consumer.CASES:
            for size in (257, 65537):
                samples = list(range(1, 18))
                cells.append({"family": name, "size": size, "passed": True, "outputs": [tensor()] * 17,
                              "inputs": [tensor()] * 23, "first": tensor(), "firstNs": 10, "factoryNs": 1,
                              "samples": samples, "statistics": consumer.summary(samples),
                              "compilerExecutors": [{"options": list(consumer.COMPILER_OPTIONS), "nvrtcVersion": [13, 0]}],
                              "firstOwnerReleasedBeforeWarmup": True,
                              "warmups": [{"phase": p, "completed": True, "ownerReleased": True} for p in range(1, 6)],
                              "ownership": {"retainedOutputs": 17, "computedPointers": list(range(17)), "inputPointers": [100]}})
        records.append({"version": consumer.VERSION, "bindings": consumer.bindings(), "passed": True,
                        "index": index, "label": label, "implementation": implementation,
                        "source": {"status": "", "commit": consumer.BASELINE if label == "B" else "c" * 40,
                                   "tree": "e" * 40, "files": {"source": "a" * 64}},
                        "buildRecordSha256": "b" * 64,
                        "installed": {"wheelSha256": "d" * 64, "files": {"torch_rs/__init__.py": "a" * 64}},
                        "interpreterSha256": "e" * 64, "frameworkSha256": "a" * 64,
                        "libraries": [{"path": "/owned/libcudart.so.13", "sha256": "e" * 64, "version": 13000},
                                      {"path": "/owned/libnvrtc.so.13", "sha256": "d" * 64, "version": [13, 0]}],
                        "compiler": {"path": "/owned/libnvrtc.so.13", "sha256": "d" * 64, "version": [13, 0]},
                        "environment": {"TORCH_RS_CUDART": "/owned/libcudart.so.13", "TORCH_RS_NVRTC": "/owned/libnvrtc.so.13"},
                        "frameworkVersion": "2.13.0+cu130", "referenceUuid": consumer.GPU,
                        "referenceExtension": {"sha256": "a" * 64},
                        "freezeSha256": "f" * 64, "preflightSha256": "a" * 64, "gpu": consumer.GPU, "gpuBefore": "snapshot", "gpuAfter": "snapshot",
                        "synchronization": {"api": "cudaDeviceSynchronize", "logicalDevice": 0,
                                            "path": "/owned/libcudart.so.13", "sha256": "e" * 64, "version": 13000,
                                            "runtimeIdentity": {"visible_count": 1, "logical_index": 0, "pci.bus_id": "0000:01:00.0",
                                                                "uuid": consumer.GPU, "physical_index": 0,
                                                                "uuid_source": "physical inventory matched by runtime PCI bus ID"},
                                            "physicalInventory": [{"index": 0, "uuid": consumer.GPU, "pci.bus_id": "00000000:01:00.0"}]},
                        "startedAt": str(index), "finishedAt": str(index), "cells": cells,
                        "churn": {"calls": [{"size": size, "elapsedNs": 1, "input": tensor(), "output": tensor()}
                                             for size in consumer.LENGTHS]}})
        if (label, implementation) == ("C", "native"):
            records[-1]["controls"] = {
                "timingData": False,
                "matrix": [{"family": name, "size": size, "companionInvocation": True,
                            "selected": selection("vm", 321) if name == "over_cap" else selection()}
                           for name, _ in consumer.CASES for size in (257, 65537)],
                "negation": [{"size": size, "selected": selection(), "preparationObject": 1 if i in (1, 2) else i + 10,
                              "executorObject": 10, "preparationKey": str(size), "hit": i == 2,
                              "counts": {"hostPlan": 0 if i == 2 else 1, "accounting": 0 if i == 2 else 1}}
                             for i, size in enumerate(consumer.LENGTHS)],
                "broadcast": [{"width": width, "selected": selection(identity=f"{width:02x}"),
                               "executorObject": i, "retainedExecutors": min(i + 1, 8),
                               "retainedPreparations": min(i + 1, 8)}
                              for i, width in enumerate(consumer.WIDTHS)]}
    return records


class ConsumerTests(unittest.TestCase):
    def rejects(self, mutate):
        data = reports()
        mutate(data)
        self.assertFalse(consumer.verify_records(data, consumer.bindings())["passed"])

    def test_complete_ledger(self):
        result = consumer.verify_records(reports(), consumer.bindings())
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["ledger"], {"cells": 96, "samples": 1632, "warmups": 480, "firstCalls": 96})
        self.assertEqual(len(result["ratios"]), 72)

    def test_missing_leg(self):
        with self.assertRaises(ValueError):
            consumer.verify_records(reports()[:-1], consumer.bindings())

    def test_order(self):
        self.rejects(lambda data: data.reverse())

    def test_missing_sample(self):
        self.rejects(lambda data: data[0]["cells"][0]["samples"].pop())

    def test_missing_cell(self):
        self.rejects(lambda data: data[0]["cells"].pop())

    def test_failed_reference(self):
        self.rejects(lambda data: data[1].update(passed=False))

    def test_changed_freeze(self):
        self.rejects(lambda data: data[2].update(freezeSha256="different"))

    def test_dirty_source(self):
        self.rejects(lambda data: data[2]["source"].update(status=" M source"))

    def test_runtime(self):
        self.rejects(lambda data: data[1]["synchronization"].update(api="eager_synchronize"))

    def test_lifetime(self):
        self.rejects(lambda data: data[0]["cells"][0].update(firstOwnerReleasedBeforeWarmup=False))

    def test_output_alias(self):
        self.rejects(lambda data: data[0]["cells"][0]["ownership"]["computedPointers"].append(100))

    def test_nan_and_infinity_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(ValueError):
                consumer.compare(tensor(value), tensor(value))

    def test_signed_zero_exact_native(self):
        with self.assertRaises(ValueError):
            consumer.compare(tensor("0x0.0p+0"), tensor("-0x0.0p+0"), exact=True)
        consumer.compare(tensor("0x0.0p+0"), tensor("-0x0.0p+0"))

    def test_reference_tolerance(self):
        consumer.compare(tensor(float(1.00019).hex()), tensor())
        with self.assertRaises(ValueError):
            consumer.compare(tensor(float(1.00021).hex()), tensor())

    def test_metadata_exact(self):
        changed = tensor()
        changed["stride"] = [2]
        with self.assertRaises(ValueError):
            consumer.compare(tensor(), changed)

    def test_quartiles(self):
        self.assertEqual(consumer.summary(list(range(1, 18))),
                         {"median": 9, "q1": 5, "q3": 13, "mad": 4, "min": 1, "max": 17})

    def test_selected_invocation_controls_required(self):
        self.rejects(lambda data: data[2].pop("controls"))

    def test_over_cap_precondition(self):
        self.rejects(lambda data: data[2]["controls"]["matrix"][-1]["selected"].update(instructions=256))

    def test_hit_history(self):
        self.rejects(lambda data: data[2]["controls"]["negation"][2].update(preparationObject=123))

    def test_wheel_binding(self):
        self.rejects(lambda data: data[3]["installed"].update(wheelSha256="e" * 64))

    def test_changed_candidate_commit(self):
        self.rejects(lambda data: data[5]["source"].update(commit="e" * 40))

    def test_compile_owner_reuse(self):
        self.rejects(lambda data: data[2]["controls"]["negation"][-1].update(executorObject=42))

    def test_no_accounting_on_hit(self):
        self.rejects(lambda data: data[2]["controls"]["negation"][2]["counts"].update(accounting=1))

    def test_missing_retained_nvrtc(self):
        self.rejects(lambda data: data[0]["libraries"].pop())

    def test_native_compiler_bytes_mismatch(self):
        self.rejects(lambda data: data[2]["compiler"].update(sha256="a" * 64))

    def test_native_compiler_options_mismatch(self):
        self.rejects(lambda data: data[0]["cells"][0]["compilerExecutors"][0].update(options=["--use_fast_math"]))

    def test_native_compiler_version_mismatch(self):
        self.rejects(lambda data: data[0]["cells"][0]["compilerExecutors"][0].update(nvrtcVersion=[12, 6]))

    def test_runtime_selection_mismatch(self):
        self.rejects(lambda data: data[0]["environment"].update(TORCH_RS_CUDART="/different/libcudart.so.12"))

    def test_duplicate_runtime_loaded(self):
        self.rejects(lambda data: data[0]["libraries"].append(
            {"path": "/another/libcudart.so.13", "sha256": "e" * 64, "version": 13000}))

    def test_empty_runtime_preflight(self):
        directory = consumer.ROOT / "target/program-identity/library-controls"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            empty = Path(temporary) / "libcudart.so.13"
            empty.touch()
            with self.assertRaisesRegex(ValueError, "nonempty"):
                consumer.library_file(empty)
            with self.assertRaises(ValueError):
                consumer.library_file(Path(temporary))
            with self.assertRaises(ValueError):
                consumer.library_file(Path(temporary) / "missing")

    def test_runtime_pci_inventory_binding(self):
        devices = [{"index": 0, "uuid": consumer.GPU, "pci.bus_id": "00000000:01:00.0"}]
        observed = {"visible_count": 1, "logical_index": 0, "pci.bus_id": "0000:01:00.0"}
        bound = consumer.bind_runtime_identity(observed, devices)
        self.assertEqual(bound["uuid"], consumer.GPU)
        self.assertEqual(bound["uuid_source"], "physical inventory matched by runtime PCI bus ID")
        for invalid in (dict(observed, visible_count=2), dict(observed, logical_index=1),
                        dict(observed, **{"pci.bus_id": "0000:02:00.0"})):
            with self.assertRaises(ValueError):
                consumer.bind_runtime_identity(invalid, devices)
        for invalid_devices in ([], devices * 2, [dict(devices[0], uuid="GPU-wrong")],
                                [dict(devices[0], index=1)]):
            with self.assertRaises(ValueError):
                consumer.bind_runtime_identity(observed, invalid_devices)

    def test_runtime_uuid_provenance_verifier(self):
        self.rejects(lambda data: data[0]["synchronization"]["runtimeIdentity"].update(uuid="GPU-wrong"))
        self.rejects(lambda data: data[0]["synchronization"]["runtimeIdentity"].update(visible_count=2))
        self.rejects(lambda data: data[0]["synchronization"]["runtimeIdentity"].update(uuid_source="requested UUID"))

    def test_fixed_protocol_bytes(self):
        self.assertEqual(consumer.sha((PATH.parent / "protocol.md").read_bytes()), consumer.PROTOCOL_SHA)
        self.assertIn("range(37)", dict(consumer.CASES)["shared"])
        self.assertIn("range(320)", dict(consumer.CASES)["over_cap"])
        self.assertEqual(consumer.LENGTHS, (3, 5, 5, 7, 9, 11, 13, 15, 17, 19, 21, 5))


if __name__ == "__main__":
    unittest.main()
