"""Capture required regressions without mixing candidate and reference imports.

Supplemental correctness receipts only; the fixed six-case evaluator is unchanged.
Run from the composite root with its canonical .venv.
"""
import ctypes, datetime, hashlib, importlib, json, math, os, pathlib, subprocess, sys
import numpy as np
root = pathlib.Path.cwd()
out = root / 'target/postcommit-eb2af97'
sys.path.insert(0, str(root / 'scripts'))
from evaluate_cuda_math import BlockTorch, runtime_provenance
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt = json.loads((out / 'build-capture/build-record.json').read_text())

def worker(role):
    blocker = BlockTorch() if role == 'candidate' else None
    if blocker:
        sys.meta_path.insert(0, blocker)
    m = importlib.import_module('torch_rs' if blocker else 'torch')
    result = dict(role=role, pid=os.getpid(), executable=sys.executable,
                  package=str(pathlib.Path(m.__file__).resolve()), python=sys.version,
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=[])
    if blocker:
        path = pathlib.Path(m._C.__file__).resolve()
        assert sha(path) == receipt['extension_sha256']
        result['extension'] = dict(path=str(path), sha256=sha(path))
    else:
        result['reference_version'] = m.__version__
        result['gpu'] = m.cuda.get_device_name(0)
    for rows in (1, 2, 17):
        patterns = [[1e8, 1, 1, -1e8]]
        for large in (np.float32(3e38), np.finfo(np.float32).max):
            patterns += [[large, 0, large, -large], [large, -large, large, -large]]
        for pattern in patterns:
            values = np.tile(np.asarray(pattern, dtype=np.float32), (rows, 1))
            x = m.tensor(values.ravel().tolist(), dtype=m.float32).reshape(rows, 4).to('cuda:0')
            for keepdim in (False, True):
                actual = x.sum(1, keepdim).cpu().tolist()
                result['cases'].append(dict(kind='cancellation-overflow', rows=rows,
                    pattern=[float(v) for v in pattern], keepdim=keepdim, values=actual))
            np.testing.assert_array_equal(x.cpu().tolist(), values)
    for width in (65539, 1_000_000, 1_000_003):
        for value in (0.1, -0.1, 0.01, 1.1):
            x = m.full((2, width), value).to('cuda:0')
            result['cases'].append(dict(kind='wide-decimal', width=width, value=value,
                                       values=x.sum(1).cpu().tolist()))
    lib = ctypes.CDLL(os.environ['TORCH_RS_CUDART'])
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    for width in (536870912, 536870916):
        x = m.zeros((1, width), device='cuda:0')
        lib.cudaDeviceSynchronize()
        for index, value in ((513002932, 1e8), (279983480, -1e8), (6175757, -1.)):
            cell = ctypes.c_float(value)
            assert lib.cudaMemcpy(x.data_ptr() + index * 4, ctypes.byref(cell), 4, 1) == 0
        for keepdim in (False, True):
            result['cases'].append(dict(kind='indexing-boundary', width=width, keepdim=keepdim,
                                       values=x.sum(1, keepdim).cpu().tolist()))
        del x
    x = m.tensor([1e20, -20.], requires_grad=True)
    (m.nn.functional.glu(x) * 1e20).sum().backward()
    result['glu_gradient'] = x.grad.tolist()
    if blocker:
        blocker.check()
        result['blocked_imports'] = blocker.attempts
        result['loaded_torch_modules'] = [n for n in sys.modules if n == 'torch' or n.startswith('torch.')]
    result.update(runtime_provenance())
    result['ended_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    def json_values(value):
        if isinstance(value, float) and not math.isfinite(value):
            return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
        if isinstance(value, list):
            return [json_values(v) for v in value]
        if isinstance(value, dict):
            return {k:json_values(v) for k,v in value.items()}
        return value
    print(json.dumps(json_values(result), indent=2, allow_nan=False))

if len(sys.argv) > 1:
    worker(sys.argv[1])
else:
    results = {}
    for role in ('reference', 'candidate'):
        command = [sys.executable, '-I', '-B', str(pathlib.Path(__file__).resolve()), role]
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        run = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (out / f'numerical-{role}.json').write_text(run.stdout)
        (out / f'numerical-{role}.stderr.log').write_text(run.stderr)
        (out / f'numerical-{role}.command.json').write_text(json.dumps(dict(command=command,
             started_at=start, ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             exit_status=run.returncode), indent=2)+'\n')
        run.check_returncode()
        results[role] = json.loads(run.stdout)
    a, b = results['candidate'], results['reference']
    assert a['pid'] != b['pid']
    assert len(a['cases']) == len(b['cases']) == 46
    for actual, expected in zip(a['cases'], b['cases']):
        av, bv = np.asarray(actual['values'], dtype=float), np.asarray(expected['values'], dtype=float)
        assert {k:v for k,v in actual.items() if k != 'values'} == {k:v for k,v in expected.items() if k != 'values'}
        if actual['kind'] != 'wide-decimal':
            np.testing.assert_array_equal(av, bv)
        else:
            np.testing.assert_allclose(av, bv, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(a['glu_gradient'], b['glu_gradient'], rtol=2e-6, atol=1e-7)
    assert np.isfinite(a['glu_gradient']).all()
    print('46 row-sum cases passed: 30 cancellation/overflow, 4 indexing-boundary and 12 wide-decimal comparisons.')
    print('Indexing boundary outputs:', a['cases'][-4:], b['cases'][-4:])
    print('GLU native/reference gradients:', a['glu_gradient'], b['glu_gradient'])
    print('Separate processes; candidate PyTorch imports:', a['loaded_torch_modules'], a['blocked_imports'])
    print('Correctness only; nonfinite observations use explicit NaN/Infinity strings in strict JSON.')
