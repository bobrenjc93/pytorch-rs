import hashlib, json, pathlib, sys, zipfile
import torch_rs
root=pathlib.Path(__file__).resolve().parents[2]
snapshot=root/'target/python314-checkout'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert pathlib.Path(sys.prefix).resolve()==snapshot/'.venv'
package=pathlib.Path(torch_rs.__file__).resolve().parent
extension=pathlib.Path(torch_rs._C.__file__).resolve()
wheel,=list((snapshot/'target/wheels').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    assert hashlib.sha256(z.read('torch_rs/'+extension.name)).hexdigest()==sha(extension)
    files={}
    for source in (root/'python/torch_rs').rglob('*.py'):
        relative=source.relative_to(root/'python/torch_rs')
        digest=sha(source)
        assert digest==sha(package/relative)==sha(snapshot/'python/torch_rs'/relative)
        assert hashlib.sha256(z.read('torch_rs/'+str(relative))).hexdigest()==digest
        files[str(relative)]=digest
result=dict(scope='Python 3.14 precommit compatibility build; not a clean checkout',
            source_snapshot=str(snapshot),
            build_command=['.venv/bin/maturin','build','--release','--locked','--out','target/wheels'],
            build_cwd=str(snapshot),
            build_environment={'PYO3_PYTHON':str(snapshot/'.venv/bin/python'),
                               'VIRTUAL_ENV':str(snapshot/'.venv'),
                               'CARGO_TARGET_DIR':str(snapshot/'target/cargo-build')},
            profile={'name':'release','lto':'thin','codegen_units':1,'features':['extension-module']},
            build_log_sha256=sha(root/'target/integration/build314.log'),
            python=sys.version, python_executable=sys.executable,
            interpreter_path=str(pathlib.Path(sys.executable).resolve()),
            interpreter_sha256=sha(sys.executable),
            extension_path=str(extension),extension_sha256=sha(extension),
            wheel_path=str(wheel),wheel_sha256=sha(wheel),python_files=files)
(root/'target/integration/build314-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Python 3.14 installed package, extension, wheel, source snapshot and interpreter verified')
