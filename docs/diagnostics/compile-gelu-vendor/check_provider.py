"""Offline provider-input selection diagnostic; no framework or GPU execution.

Uses only the declared CUDA12.8 compiler. The empty-directory control must not
be interpreted as an alternate math provider or a numerical validation.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
INPUT_SHA = '1fc1bc8d4131d5a59a91fc26f4908886e231f842c9e83ae951979ff3df82bdb5'
NVCC = Path('/usr/local/cuda-12.8/bin/nvcc')
NVCC_SHA = '59e4e55f9a38b78c590df1e28a69dad91052958f18a03868ca4a547c043fbba7'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('all inputs and outputs must reside inside this worktree')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    library, out = inside(args.input), inside(args.output)
    assert library.name == 'libdevice.10.bc' and sha(library) == INPUT_SHA
    assert sha(NVCC) == NVCC_SHA
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent / 'selection/provider.cu'
    empty = out / 'empty-library'; empty.mkdir()
    report = dict(declared_at=stamp(), code_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source={'path':str(source), 'sha256':sha(source)},
        requested_library={'path':str(library), 'sha256':sha(library)},
        tools={str(p):sha(p) for p in (NVCC, Path('/usr/local/cuda-12.8/nvvm/bin/cicc'),
                                      Path('/usr/bin/strace'), Path('/usr/bin/g++'))},
        planned=['official-directory', 'empty-directory'], runs=[])
    (out/'declaration.json').write_text(json.dumps(report, indent=2)+'\n')
    env = dict(os.environ)
    env['PATH'] = '/usr/local/cuda-12.8/nvvm/bin:' + env.get('PATH','')
    for name, directory in [('official-directory',library.parent),('empty-directory',empty)]:
        run=out/name;run.mkdir();tmp=run/'tmp';tmp.mkdir();env['TMPDIR']=str(tmp)
        trace, ptx = run/'compiler.trace',run/'provider.ptx'
        command=['/usr/bin/strace','-f','-ttt','-s','4096','-yy','-e','trace=%file,process',
                 '-o',str(trace),str(NVCC),'--dont-use-profile','--libdevice-directory',str(directory),
                 '-I/usr/local/cuda-12.8/include','--compiler-bindir','/usr/bin/g++','--ptx',
                 '--relocatable-device-code=true','--gpu-architecture=compute_75','--ftz=true',
                 '--fmad=true','--prec-div=true','--prec-sqrt=true','--std=c++17','--verbose',
                 str(source),'-o',str(ptx)]
        record=dict(name=name,command=command,started_at=stamp(),
                    library_directory_entries=sorted(p.name for p in directory.iterdir()))
        with (run/'stdout.log').open('wb') as stdout,(run/'stderr.log').open('wb') as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
            record['pid']=process.pid
            record['exit_code']=process.wait()
        record['child_returned_at']=stamp()
        text=trace.read_text()
        # Report every file syscall mentioning libdevice, including failures.
        record['libdevice_file_accesses']=[line for line in text.splitlines()
             if 'libdevice' in line and any(token in line for token in
                ('open(', 'openat(', 'openat2(', 'stat(', 'statx(', 'newfstatat(', 'access('))]
        record['trace_sha256']=sha(trace)
        (run/'compiler.trace.gz').write_bytes(gzip.compress(trace.read_bytes(),mtime=0))
        trace.unlink()
        record['ptx_sha256']=sha(ptx) if ptx.exists() else None
        report['runs'].append(record)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['identical_ptx']=(out/'official-directory/provider.ptx').read_bytes()==(out/'empty-directory/provider.ptx').read_bytes()
    report['requested_input_unchanged']=sha(library)==INPUT_SHA
    report['provider_selection_established']=False
    report['finished_at']=stamp()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'exits':[r['exit_code'] for r in report['runs']],
                      'file_accesses':[len(r['libdevice_file_accesses']) for r in report['runs']],
                      'identical_ptx':report['identical_ptx']}))


if __name__=='__main__':main()
