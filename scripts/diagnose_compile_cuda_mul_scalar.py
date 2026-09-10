#!/usr/bin/env python3
"""mul_neg_add_v1: non-scoring correctness evidence, independent of frozen corpora."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from . import diagnose_compile_cuda_add as frozen
from . import diagnose_compile_cuda_neg_add as neg_add
from tests.test_compile_cuda_mul_scalar import generated_source, make_program

CASE_SET = 'mul_neg_add_v1'
GRAMMAR_SEED = 791406


def scale_left(x):
    return -1.375 * x


def scale_method(x):
    return x.multiply(0.375)


def composite(x, y):
    return -(x * -0.5).add(y.mul(1.375)) * 2


def cases(mask):
    if mask == '0,1':
        return [(p, (19,), arity, 'ordinal1', True) for p, arity in
                ((scale_left, 1), (scale_method, 1), (composite, 2))] + [
                    (composite, (19,), 2, 'mixed_ordinals', False)], {}
    rng = frozen.np.random.default_rng(GRAMMAR_SEED)
    sources = [generated_source(rng, length) for length in (1, 3, 9, 23)]
    generated = []
    source_records = {}
    for i, source in enumerate(sources):
        program = make_program(source)
        program.__name__ = f'generated_{i}'
        generated.append((program, 2))
        source_records[program.__name__] = {
            'source': source, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
        }
    shapes = [(), (0,), (2, 0, 3), (257,), (11, 37), (65539,)]
    result = [(p, shape, arity, 'offset', True)
              for p, arity in ((scale_left, 1), (scale_method, 1), (composite, 2), *generated)
              for shape in shapes]
    result += [(composite, (3, 4), 2, kind, False) for kind in
               ('transpose', 'broadcast', 'gradient', 'float64', 'mixed_cpu')]
    result += [(p, (3, 4), 1, 'plain', False) for p in
               (frozen.reject_multiply, frozen.reject_scalar, frozen.reject_detach,
                frozen.reject_float, frozen.reject_reduce)]
    return result, source_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-set', choices=[CASE_SET], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = {'schema': CASE_SET, 'purpose': 'non-scoring correctness only; no timing or parity claim',
              'reference_backend': 'eager', 'grammar_seed': GRAMMAR_SEED,
              'source_attribution': 'Working tree on base_commit; source_sha256 identifies the measured source.',
              'input_seeds': [frozen.SEED, frozen.SEED + 1], 'cases': []}
    try:
        mask = os.environ.get('CUDA_VISIBLE_DEVICES')
        report.update(neg_add.environment(mask))
        root = neg_add.ROOT
        paths = [*sorted((root / 'src').rglob('*.rs')), *sorted((root / 'src').rglob('*.ptx')),
                 root / 'Cargo.toml', root / 'Cargo.lock', root / 'pyproject.toml', root / 'uv.lock',
                 Path(__file__), root / 'tests/test_compile_cuda_mul_scalar.py']
        report['source_sha256'].update({str(p.relative_to(root)): frozen.sha(p) for p in paths})
        report['environment']['nvcc_available'] = frozen.command('nvcc', '--version')
        report['environment'].update(
            python_full=sys.version, python_build=platform.python_build(),
            python_compiler=platform.python_compiler(),
            packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        )
        matrix, report['generated_programs'] = cases(mask)
        for program, shape, arity, kind, supported in matrix:
            for fullgraph in (True, False):
                record = frozen.run_case(program, shape, arity, kind, fullgraph, supported)
                record['expectation_met'] = neg_add.expectation_met(record)
                report['cases'].append(record)
    except Exception as error:
        report['setup_error'] = f'{type(error).__name__}: {error}'
    records = report['cases']
    report['summary'] = {
        'cases': len(records),
        'reference_eligible': sum(r['reference_eligible'] for r in records),
        'native_pass': sum(r['native_outcome'] == 'pass' for r in records),
        'native_unsupported': sum(r['native_outcome'] == 'unsupported' for r in records),
        'expectation_failures': sum(not r['expectation_met'] for r in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))
    return int(not records or 'setup_error' in report or report['summary']['expectation_failures'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
