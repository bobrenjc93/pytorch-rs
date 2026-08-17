import copy
import importlib
import inspect
import pickle
import threading
import types
import unittest

import torch_rs as torch

from torch_rs.utils.data import get_worker_info


FUNCTION_DOC = """Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """


class GetWorkerInfoTests(unittest.TestCase):
    def test_main_process_and_ordinary_threads_return_none_without_state(self):
        worker_module = importlib.import_module(
            "torch_rs.utils.data._utils.worker"
        )
        self.assertNotIn("_worker_info", worker_module.__dict__)
        self.assertNotIn("WorkerInfo", worker_module.__dict__)

        for _ in range(3):
            self.assertIsNone(get_worker_info())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [object()] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    get_worker_info(),
                    torch.utils.data.get_worker_info(),
                )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [(None, None)] * worker_count)
        self.assertNotIn("_worker_info", worker_module.__dict__)
        self.assertNotIn("WorkerInfo", worker_module.__dict__)

    def test_signature_annotations_documentation_and_metadata(self):
        function = get_worker_info
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> 'WorkerInfo | None'")
        self.assertEqual(function.__annotations__, {"return": "WorkerInfo | None"})
        self.assertEqual(function.__module__, "torch_rs.utils.data._utils.worker")
        self.assertEqual(function.__name__, "get_worker_info")
        self.assertEqual(function.__qualname__, "get_worker_info")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        cases = (
            (
                lambda: get_worker_info(None),
                "get_worker_info() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: get_worker_info(None, None),
                "get_worker_info() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: get_worker_info(value=None),
                "get_worker_info() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: get_worker_info(None, value=None),
                "get_worker_info() got an unexpected keyword argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_public_export_private_owner_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")
        worker_module = importlib.import_module(
            "torch_rs.utils.data._utils.worker"
        )

        self.assertIs(torch.utils.data.get_worker_info, get_worker_info)
        self.assertIs(data_module.get_worker_info, get_worker_info)
        self.assertIs(worker_module.get_worker_info, get_worker_info)
        self.assertNotIn("get_worker_info", dataset_module.__dict__)
        self.assertEqual(data_module.__all__.count("get_worker_info"), 1)

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["get_worker_info"], get_worker_info)

        self.assertFalse(hasattr(data_module, "DataLoader"))
        self.assertFalse(hasattr(data_module, "WorkerInfo"))
        self.assertFalse(hasattr(worker_module, "WorkerInfo"))
        self.assertFalse(hasattr(worker_module, "_worker_info"))

    def test_pickle_and_copy_preserve_the_global_function(self):
        self.assertIs(copy.copy(get_worker_info), get_worker_info)
        self.assertIs(copy.deepcopy(get_worker_info), get_worker_info)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(get_worker_info, protocol=protocol)
                )
                self.assertIs(restored, get_worker_info)


if __name__ == "__main__":
    unittest.main()
