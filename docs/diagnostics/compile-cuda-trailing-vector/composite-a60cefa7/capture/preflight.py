import importlib.metadata as md, json, pathlib, platform, sys
import numpy, torch, torch_rs
root = pathlib.Path.cwd().resolve()
assert pathlib.Path(sys.prefix).resolve() == root / '.venv'
assert not (root / '.venv').is_symlink()
for module in (numpy, torch, torch_rs):
    pathlib.Path(module.__file__).resolve().relative_to(root / '.venv')
assert torch.__version__.split('+')[0] == '2.13.0'
assert torch.cuda.is_available()
print(json.dumps(dict(python=sys.version, executable=sys.executable, compiler=platform.python_compiler(), build=platform.python_build(), prefix=sys.prefix, numpy_version=numpy.__version__, numpy_file=numpy.__file__, torch_version=torch.__version__, torch_file=torch.__file__, torch_cuda=torch.version.cuda, torch_rs_version=md.version('torch-rs'), torch_rs_file=torch_rs.__file__, gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count()), indent=2))
