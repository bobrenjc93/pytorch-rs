import json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from evaluate_cuda_math import sha256,source_provenance
import torch_rs
record=json.loads(Path(sys.argv[1]).read_text())
assert all(record[k] == source_provenance()[k] for k in source_provenance())
assert sha256(torch_rs._C.__file__)==record['extension_sha256']
installed=Path(torch_rs.__file__).parent
files={}
for source in (Path.cwd()/'python/torch_rs').rglob('*.py'):
    relative=source.relative_to(Path.cwd()/'python/torch_rs')
    assert sha256(source)==sha256(installed/relative),relative
    files[str(relative)]=sha256(source)
with zipfile.ZipFile(record['wheel_path']) as wheel:
    import hashlib
    for name,digest in files.items():
        assert hashlib.sha256(wheel.read('torch_rs/'+name)).hexdigest()==digest,name
print(json.dumps({'verified_python_files':len(files),'native_sha256':sha256(torch_rs._C.__file__),'source':source_provenance(),'wheel_sha256':sha256(record['wheel_path'])},indent=2))
