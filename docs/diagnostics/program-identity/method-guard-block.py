"""Source-extracted guard controls and block timing, never public-call latency."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = 'python/torch_rs/_compile_pointwise.py'
BEFORE = '6f0123a2cc521af9fab3dc19f1c256b28684f0db'
REUSED = '9009c8ea6de40ec55c970983602c8fadd8a0447a'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def extract_blocks(source):
    tree = ast.parse(source)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '_METHOD_GUARDS' for t in n.targets)]
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Name) and n.iter.id == '_METHOD_GUARDS']
    assert len(assignments) == len(loops) == 1
    return assignments[0], loops[0]


def make_guard(source, root, methods, missing, unsupported, instrument=False):
    assignment, loop = copy.deepcopy(extract_blocks(source))
    counts = {'dictionaryFetches': 0, 'comparisons': 0, 'order': []}

    def fetch(owner):
        counts['dictionaryFetches'] += 1
        return owner.__dict__

    def different(value, expected, owner, name):
        counts['comparisons'] += 1
        counts['order'].append((owner, name))
        return value is not expected

    class Instrument(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if node.attr == '__dict__':
                return ast.copy_location(ast.Call(ast.Name('_fetch', ast.Load()), [node.value], []), node)
            return self.generic_visit(node)

        def visit_Compare(self, node):
            node = self.generic_visit(node)
            assert len(node.ops) == 1 and isinstance(node.ops[0], ast.IsNot)
            return ast.copy_location(ast.Call(ast.Name('_different', ast.Load()),
                [node.left, node.comparators[0], ast.Name('cls', ast.Load()), ast.Name('name', ast.Load())], []), node)

    if instrument:
        loop = Instrument().visit(loop)
    namespace = {'_ROOT': root, '_METHODS': methods, '_MISSING': missing,
                 'unsupported': unsupported, '_fetch': fetch, '_different': different}
    exec(compile(ast.fix_missing_locations(ast.Module([assignment], [])), '<guard-specification>', 'exec'), namespace)
    wrapper = ast.parse('def guard():\n    pass\n')
    wrapper.body[0].body = [loop]
    exec(compile(ast.fix_missing_locations(wrapper), '<extracted-guard-block>', 'exec'), namespace)
    return namespace['guard'], namespace['_METHOD_GUARDS'], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, wheel = args.output.resolve(), args.wheel.resolve()
    assert output.is_relative_to(ROOT) and wheel.is_relative_to(ROOT)
    output.parent.mkdir(parents=True, exist_ok=False)
    record = {'kind': 'source-extracted-guard-block-diagnosis', 'passed': False,
              'ordinaryCallLatencyOrScore': False, 'command': sys.orig_argv,
              'startedAt': datetime.now(timezone.utc).isoformat(), 'head': git('rev-parse', 'HEAD'),
              'status': git('status', '--short'), 'scriptSha256': digest(Path(__file__).read_bytes())}
    try:
        import torch_rs as native
        from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
        package = Path(native.__file__).resolve().parent
        assert package.is_relative_to(ROOT) and Path(sys.executable).resolve().is_relative_to(ROOT)
        with zipfile.ZipFile(wheel) as archive:
            installed = {n: archive.read(n) for n in archive.namelist()
                         if n.startswith('torch_rs/') and n.endswith(('.py', '.so'))}
        for name, data in installed.items():
            assert (package.parent / name).read_bytes() == data
            if name.endswith('.py'):
                assert (ROOT / 'python' / name).read_bytes() == data
        sources = {label: git('show', f'{commit}:{FRONTEND}') for label, commit in
                   (('before', BEFORE), ('grouped', REUSED))}
        assert (ROOT / FRONTEND).read_text().strip() == sources['before']
        owners = (native.Tensor, native.Tensor.__base__)
        assert all(type(owner) is type for owner in owners)
        record.update(interpreter=str(Path(sys.executable).resolve()), python=sys.version,
                      interpreterSha256=digest(Path(sys.executable).read_bytes()),
                      package=str(package), wheel=str(wheel), wheelSha256=digest(wheel.read_bytes()),
                      installedHashes={n: digest(d) for n, d in installed.items()},
                      nativeExtension=str(Path(bridge.__file__).resolve()),
                      nativeExtensionSha256=digest(Path(bridge.__file__).read_bytes()),
                      owners=[{'module': o.__module__, 'name': o.__qualname__, 'object': id(o),
                               'metaclass': type(o).__name__, 'namespaceType': type(o.__dict__).__name__} for o in owners],
                      sources={label: {'commit': BEFORE if label == 'before' else REUSED,
                          'sourceBlobSha256': digest(subprocess.check_output(
                              ['git', 'show', f'{BEFORE if label == "before" else REUSED}:{FRONTEND}'], cwd=ROOT)),
                          'normalizedSourceSha256': digest(text.encode()),
                          'blocks': [ast.unparse(n) for n in extract_blocks(text)]} for label, text in sources.items()},
                      controls={}, timings=[], iterationsPerSample=20000,
                      samplesPerArm=17, warmupBatches=5, warmupIterations=1000,
                      order=['before', 'grouped', 'grouped', 'before'], gcEnabled=gc.isenabled())
        functions, tables = {}, {}
        for label, text in sources.items():
            fn, table, counts = make_guard(text, native, frontend._METHODS, frontend._MISSING,
                                          frontend.unsupported, instrument=True)
            fn()
            record['controls'][label] = {**counts, 'order': [(owners.index(o), n) for o, n in counts['order']]}
            assert counts['comparisons'] == 38
            assert counts['dictionaryFetches'] == (38 if label == 'before' else 2)
            functions[label], tables[label], _ = make_guard(text, native, frontend._METHODS,
                                                           frontend._MISSING, frontend.unsupported)
        flat = tuple((owner, name, expected) for owner, guards in tables['grouped'] for name, expected in guards)
        assert len(flat) == len(tables['before']) == 38
        assert all(a[0] is b[0] and a[1] == b[1] and a[2] is b[2]
                   for a, b in zip(flat, tables['before']))
        for label in record['order']:
            fn = functions[label]
            for _ in range(record['warmupBatches']):
                for _ in range(record['warmupIterations']):
                    fn()
            samples = []
            for _ in range(record['samplesPerArm']):
                start = time.perf_counter_ns()
                for _ in range(record['iterationsPerSample']):
                    fn()
                samples.append(time.perf_counter_ns() - start)
            record['timings'].append({'arm': label, 'elapsedNs': samples,
                'medianNsPerBlock': statistics.median(samples) / record['iterationsPerSample']})
        assert digest(Path(bridge.__file__).read_bytes()) == record['nativeExtensionSha256']
        record['passed'] = True
    except BaseException:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record['finishedAt'] = datetime.now(timezone.utc).isoformat()
        with output.open('xb') as stream:
            stream.write(gzip.compress((json.dumps(record, allow_nan=False) + '\n').encode(), mtime=0))


if __name__ == '__main__':
    main()
