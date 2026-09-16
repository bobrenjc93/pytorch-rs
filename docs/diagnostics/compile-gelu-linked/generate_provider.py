"""Private, host-only libNVVM input-selection probe; never imports a framework."""

import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


# Offline maintainer tool. All writable paths are rooted in this checkout.
ROOT = Path(__file__).resolve().parents[3]
NOTES = ROOT / 'target/gelu-linked/generation'
LIBRARY = Path('/usr/local/cuda-12.8/nvvm/lib64/libnvvm.so.4.0.0')
HEADER = Path('/usr/local/cuda-12.8/nvvm/include/nvvm.h')
BITCODE = ROOT / 'target/gelu-linked/input/libdevice.10.bc'
WRAPPER = ROOT / 'src/cuda/erf_provider.ll'
OPTIONS = ['-arch=compute_75', '-opt=3', '-ftz=1', '-fma=1',
           '-prec-div=1', '-prec-sqrt=1']
PINNED = {
    LIBRARY: 'dfb07dc65ffb0f7a16bc82ee1f87d7f4634b908d30aa8103d1e28afb1fe6589e',
    HEADER: '900932e7435799fab1499cd4eebe0243279ec7164cd4e646124a106b833ee47d',
    BITCODE: '1fc1bc8d4131d5a59a91fc26f4908886e231f842c9e83ae951979ff3df82bdb5',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': digest(data)}


def validate_inputs(declaration=None):
    for path, expected in PINNED.items():
        if digest(path.read_bytes()) != expected:
            raise RuntimeError('Input hash mismatch: ' + str(path))
    paths = [*PINNED, WRAPPER, Path(__file__).resolve(), Path(sys.executable).resolve()]
    values = [identity(path) for path in paths]
    if declaration is not None and values != declaration['inputs']:
        raise RuntimeError('Declared inputs changed')
    return values


def child(directory, leg):
    declaration = json.loads((directory / 'declaration.json').read_text())
    validate_inputs(declaration)
    output = directory / leg
    output.mkdir(exist_ok=False)
    report = {'leg': leg, 'startedAt': now(), 'events': [], 'modules': [],
              'options': OPTIONS, 'compilerResult': None, 'error': None}
    program = C.c_void_p()
    program_created = False
    buffers = []
    library = None

    def record():
        save(output / 'report.json', report)

    def call(name, *args, checked=True):
        status = int(getattr(library, name)(*args))
        report['events'].append({'api': name, 'status': status, 'returnedAt': now()})
        record()
        if checked and status != 0:
            raise RuntimeError(name + ' returned ' + str(status))
        return status

    def result_bytes(size_api, get_api, filename):
        size = C.c_size_t()
        call(size_api, program, C.byref(size))
        if not 0 < size.value <= 16 * 1024 * 1024:
            raise RuntimeError('Unexpected compiler buffer size: ' + str(size.value))
        buffer = C.create_string_buffer(size.value)
        call(get_api, program, buffer)
        raw = bytes(buffer.raw)
        if not raw.endswith(b'\0'):
            raise RuntimeError('Compiler output is missing its documented trailing NUL')
        text = raw[:-1]
        (output / (filename + '.api-bytes')).write_bytes(raw)
        (output / filename).write_bytes(text)
        return {'apiBytesIncludingNul': len(raw), 'apiSha256IncludingNul': digest(raw),
                'textBytesExcludingNul': len(text), 'textSha256ExcludingNul': digest(text)}

    record()
    try:
        library = C.CDLL(str(LIBRARY))
        signatures = {
            'nvvmVersion': [C.POINTER(C.c_int), C.POINTER(C.c_int)],
            'nvvmIRVersion': [C.POINTER(C.c_int)] * 4,
            'nvvmCreateProgram': [C.POINTER(C.c_void_p)],
            'nvvmDestroyProgram': [C.POINTER(C.c_void_p)],
            'nvvmAddModuleToProgram': [C.c_void_p, C.c_char_p, C.c_size_t, C.c_char_p],
            'nvvmLazyAddModuleToProgram': [C.c_void_p, C.c_char_p, C.c_size_t, C.c_char_p],
            'nvvmCompileProgram': [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)],
            'nvvmGetProgramLogSize': [C.c_void_p, C.POINTER(C.c_size_t)],
            'nvvmGetProgramLog': [C.c_void_p, C.c_char_p],
            'nvvmGetCompiledResultSize': [C.c_void_p, C.POINTER(C.c_size_t)],
            'nvvmGetCompiledResult': [C.c_void_p, C.c_char_p],
        }
        for name, arguments in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, C.c_int
        version = [C.c_int() for _ in range(2)]
        ir_version = [C.c_int() for _ in range(4)]
        call('nvvmVersion', *[C.byref(x) for x in version])
        call('nvvmIRVersion', *[C.byref(x) for x in ir_version])
        report['version'] = [x.value for x in version]
        report['irVersion'] = [x.value for x in ir_version]
        if report['version'] != [2, 0] or report['irVersion'] != [2, 0, 3, 2]:
            raise RuntimeError('Unexpected compiler API/IR version')
        create_status = int(library.nvvmCreateProgram(C.byref(program)))
        program_created = create_status == 0 and bool(program.value)
        report['events'].append({'api': 'nvvmCreateProgram', 'status': create_status,
                                 'returnedAt': now()})
        record()
        if create_status != 0:
            raise RuntimeError('nvvmCreateProgram returned ' + str(create_status))
        if not program.value:
            raise RuntimeError('Successful program creation returned null')
        modules = [('nvvmAddModuleToProgram', WRAPPER)]
        if leg == 'official-library':
            modules.append(('nvvmLazyAddModuleToProgram', BITCODE))
        for api, path in modules:
            data = path.read_bytes()
            buffer = C.create_string_buffer(data)
            buffers.append(buffer)
            report['modules'].append({'api': api, 'path': str(path),
                                      'submittedBytes': len(data),
                                      'submittedSha256': digest(bytes(buffer.raw[:len(data)]))})
            record()
            call(api, program, buffer, len(data), path.name.encode('ascii'))
        options = (C.c_char_p * len(OPTIONS))(*[x.encode('ascii') for x in OPTIONS])
        report['compileStartedAt'] = now()
        record()
        report['compileStatus'] = call('nvvmCompileProgram', program, len(OPTIONS),
                                       options, checked=False)
        report['compilerLog'] = result_bytes('nvvmGetProgramLogSize',
                                             'nvvmGetProgramLog', 'compiler.log')
        if report['compileStatus'] == 0:
            report['compilerResult'] = result_bytes('nvvmGetCompiledResultSize',
                                                    'nvvmGetCompiledResult', 'provider.ptx')
    except BaseException:
        report['error'] = traceback.format_exc()
    finally:
        if program_created and library is not None:
            try:
                report['destroyStatus'] = call('nvvmDestroyProgram', C.byref(program),
                                               checked=False)
                report['programNullAfterDestroy'] = program.value is None
            except BaseException:
                report['destroyError'] = traceback.format_exc()
        report['completedAt'] = now()
        record()
    # A compiler rejection is retained data; unexpected harness/cleanup failures
    # return nonzero. Parent interpretation never uses process exit as parity.
    return 0 if (report['error'] is None and report.get('destroyStatus') == 0) else 2


def run():
    inputs = validate_inputs()
    NOTES.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='gelu-libnvvm-generation.', dir=NOTES))
    declaration = {'declaredAt': now(), 'inputs': inputs, 'options': OPTIONS,
                   'legs': ['official-library', 'no-library'], 'timeoutSecondsPerLeg': 45,
                   'frameworkExecution': False, 'gpuExecution': False,
                   'purpose': 'Offline input selection only; no numerical or deployment claim'}
    save(directory / 'declaration.json', declaration)
    print(json.dumps({'directory': str(directory), 'declaredAt': declaration['declaredAt']}), flush=True)
    report = {'startedAt': now(), 'directory': str(directory), 'legs': []}
    for leg in declaration['legs']:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child',
                   str(directory), leg]
        row = {'leg': leg, 'command': command, 'startedAt': now()}
        try:
            process = subprocess.run(command, capture_output=True, timeout=45)
            row.update(returncode=process.returncode, timedOut=False)
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as error:
            row.update(returncode=None, timedOut=True)
            stdout, stderr = error.stdout or b'', error.stderr or b''
        except OSError as error:
            row.update(returncode=None, timedOut=False,
                       spawnError=type(error).__name__ + ': ' + str(error))
            stdout, stderr = b'', str(error).encode('utf8')
        (directory / (leg + '.stdout')).write_bytes(stdout)
        (directory / (leg + '.stderr')).write_bytes(stderr)
        row.update(completedAt=now(), stdoutSha256=digest(stdout), stderrSha256=digest(stderr))
        report['legs'].append(row)
        save(directory / 'process-report.json', report)
        print(json.dumps(row), flush=True)
    try:
        validate_inputs(declaration)
        report['inputsUnchanged'] = True
    except BaseException:
        report['inputsUnchanged'] = False
        report['inputError'] = traceback.format_exc()
    report['completedAt'] = now()
    report['experimentProcessesCompleted'] = (report['inputsUnchanged'] and
        all(row['returncode'] == 0 and not row['timedOut'] for row in report['legs']))
    report['selectionConclusion'] = 'Requires inspection of both API logs and complete emitted PTX'
    save(directory / 'process-report.json', report)
    return 0 if report['experimentProcessesCompleted'] else 2


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--run':
        raise SystemExit(run())
    if len(sys.argv) == 4 and sys.argv[1] == '--child' and sys.argv[3] in ('official-library', 'no-library'):
        output_directory = Path(sys.argv[2]).resolve()
        if output_directory.parent != NOTES or not output_directory.name.startswith('gelu-libnvvm-generation.'):
            raise SystemExit('Unexpected output directory')
        raise SystemExit(child(output_directory, sys.argv[3]))
    raise SystemExit('Use --run for a fresh private host-only experiment')
