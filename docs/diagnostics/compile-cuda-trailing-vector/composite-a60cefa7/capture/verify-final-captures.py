import hashlib, io, json, pathlib, subprocess, tarfile
root = pathlib.Path.cwd().resolve()
head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
archive = subprocess.check_output(['git','archive',head])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    committed = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile()}
paths = [root/'target/integration-evidence/trailing-single-312.json', root/'target/integration-evidence/trailing-two-device-312.json', root/'target/python314/target/integration-evidence/trailing-single-314.json', root/'target/python314/target/integration-evidence/trailing-two-device-314.json']
for path in paths:
    report = json.loads(path.read_text())
    build = report['build_record']
    assert report['base_commit'] == head == build['base_commit'] == build['measured_code_commit']
    assert report['worktree_status'] == build['origin_status'] == ''
    assert build['source_matches_commit'] and build['source_kind'] == 'clean commit export'
    assert committed == build['source_files']
    assert hashlib.sha256(json.dumps(committed, sort_keys=True).encode()).hexdigest() == build['source_manifest_sha256']
    for key in ('source_root','installed_native','installed_package','wheel','build_target'):
        pathlib.Path(build[key]).resolve().relative_to(root)
    for file_key, hash_key in (('installed_native','installed_native_sha256'),('wheel','wheel_sha256')):
        assert hashlib.sha256(pathlib.Path(build[file_key]).read_bytes()).hexdigest() == build[hash_key]
    for name, digest in report['source_sha256'].items():
        assert committed[name] == digest
    assert report['native_extension_sha256'] == build['installed_native_sha256']
    assert report['summary']['expectation_failures'] == 0
    print(path.relative_to(root), json.dumps(report['summary']))
    print('Verified clean commit, complete source manifest, harness hashes, installed native/wheel hashes and contained source/build/import paths.')
print('Measured implementation:', head)
