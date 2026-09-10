import os, pathlib, runpy, sys
root = pathlib.Path(__file__).resolve().parents[2]
snapshot = root / 'target/python314-checkout'
assert pathlib.Path(sys.prefix).resolve() == snapshot / '.venv'
os.chdir(snapshot)
sys.path.insert(0, str(snapshot))
print('compatibility cwd:', snapshot, 'interpreter:', sys.executable, flush=True)
runpy.run_module('unittest', run_name='__main__')
