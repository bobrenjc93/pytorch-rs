import hashlib, json, pathlib, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[2]
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
report=json.loads((root/'target/integration/cuda-math.json').read_text())
assert report['source_unchanged_during_run']
assert report['accounting']['denominator']==6 and report['accounting']['passed']==5
assert report['build_record']['measurement_kind']=='precommit-diagnostic'
assert report['build_record']['clean_checkout'] is False
sys.path.insert(0,str(root/'scripts'))
from evaluate_cuda_math import source_provenance
assert report['source']==source_provenance()
assert report['evaluator_sha256']==sha(root/'scripts/evaluate_cuda_math.py')
assert report['matrix_sha256']==sha(root/'docs/hardware-heterogeneity-matrix-v1.json')
paths=set()
worker_files=set()
for trial in report['trials']:
    assert trial['candidate']['pid'] != trial['reference']['pid']
    assert trial['candidate']['loaded_torch_modules']==[]
    assert trial['candidate']['blocked_imports']==[]
    assert trial['candidate']['extension']['sha256']==report['build_record']['extension_sha256']
    for role in ('candidate','reference'):
        worker=trial[role]
        for key in ('package','executable'):
            p=pathlib.Path(worker[key]).absolute()
            assert p.is_relative_to(root),p
            worker_files.add(p)
        assert worker['cuda_runtimes']
        # All actual runtime paths in both roles must be local.
        for runtime in worker['cuda_runtimes']:
            p=pathlib.Path(runtime['path']).resolve()
            assert p.is_relative_to(root),p
            paths.add(p)
assert paths
info=dict(measurement_kind='precommit-diagnostic', report_sha256=sha(root/'target/integration/cuda-math.json'),
          worker_file_hashes={str(p):sha(p) for p in sorted(worker_files)},
          runtime_hashes={str(p):sha(p) for p in sorted(paths)},
          current_source_unchanged=True, evaluator_unchanged=True,
          no_performance_credit=True)
(root/'target/integration/evaluation-audit.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
