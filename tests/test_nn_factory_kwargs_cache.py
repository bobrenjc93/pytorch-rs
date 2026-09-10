"""Exercise factory contracts across hash seeds and code-cache reconstruction."""

import os
import subprocess
import sys
import unittest

from tests.test_nn_factory_kwargs_reference import reference_torch


# marshal is the code-object serialization used in .pyc files. Keep the
# roundtrip in memory so this test never edits installed packages' caches.
_CACHE_PROBE = r"""
import inspect
import itertools
import json
import marshal
import unittest
from unittest.mock import patch

from tests import test_nn_factory_kwargs_reference as checks


def variants(function):
    code = compile(inspect.getsource(function), '<factory-source>', 'exec')
    result = {'installed': function}
    for state, compiled in (
        ('source', code),
        ('bytecode', marshal.loads(marshal.dumps(code))),
    ):
        namespace = {}
        exec(compiled, namespace)
        result[state] = namespace['factory_kwargs']
    return result


actual_variants = variants(checks.nn.factory_kwargs)
reference_variants = variants(checks.reference_nn.factory_kwargs)
values = {'device': object(), 'dtype': object(), 'memory_format': object()}
orders = {}
for state, function in reference_variants.items():
    result = function(values)
    assert result == values
    assert all(result[key] is value for key, value in values.items())
    orders[state] = list(result)
print(json.dumps(orders), flush=True)

# Metadata and reload tests still exercise the installed modules in the normal
# suite. Only behavioral tests apply to these standalone function copies.
methods = (
    'test_canonical_values_freshness_and_input_preservation_match',
    'test_errors_and_call_validation_match',
    'test_mapping_access_and_failure_order_match',
)
failed = False
for (actual_state, actual), (reference_state, reference) in itertools.product(
    actual_variants.items(), reference_variants.items()
):
    print(f'actual={actual_state}, reference={reference_state}', flush=True)
    with patch.object(checks.nn, 'factory_kwargs', actual), patch.object(
        checks.reference_nn, 'factory_kwargs', reference
    ):
        suite = unittest.TestSuite(
            checks.FactoryKwargsReferenceTests(method) for method in methods
        )
        result = unittest.TextTestRunner().run(suite)
        failed |= not result.wasSuccessful()
raise SystemExit(failed)
"""


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FactoryKwargsCacheTests(unittest.TestCase):
    def test_contracts_across_hash_seeds_and_code_caches(self):
        # Do not require an order difference on every interpreter: it is an
        # incidental set-layout effect, not a promised reference behavior.
        for seed in range(12):
            with self.subTest(seed=seed):
                completed = subprocess.run(
                    [sys.executable, "-c", _CACHE_PROBE],
                    env={**os.environ, "PYTHONHASHSEED": str(seed)},
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )


if __name__ == "__main__":
    unittest.main()
