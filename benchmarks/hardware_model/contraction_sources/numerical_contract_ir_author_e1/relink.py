"""Relink retained contract-disabled IR with explicit final FMA options."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import traceback

from numba_cuda_mlir.linker import Linker


def asset(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check(record):
    path = Path(record['path'])
    if asset(path)['sha256'] != record['sha256']:
        raise ValueError('Changed input ' + str(path))
    return path


def run(capture_path, output):
    captured = json.loads(capture_path.read_text())
    output.mkdir(exist_ok=False)
    receipt = dict(status='STARTED', source=asset(__file__),
                   capture=asset(capture_path), records=[],
                   scope='Offline relink only; no context or GPU launch')
    try:
        for row in captured['records']:
            source = row['source']
            original = check(row['cubin'])
            ir_path = check(row['cached_ltoir'])
            for fma in (True, False):
                folder = output / ('source_' + source + '_fma_' + str(fma))
                folder.mkdir()
                options = dict(cc=(8, 9), arch='sm_89', lto=True,
                               optimization_level=3, ftz=True,
                               prec_div=False, prec_sqrt=False, fma=fma,
                               debug=False, lineinfo=False,
                               max_registers=None, ptxas_options=None)
                linker = Linker(**options)
                linker.add_ltoir(ir_path.read_bytes())
                if len(linker._object_codes) != 1:
                    raise ValueError('Unexpected extra link input')
                rendered = linker._get_linker_options(False)._prepare_nvjitlink_options()
                assert '-fma=' + ('1' if fma else '0') in rendered
                cubin = folder / 'kernel.cubin'
                cubin.write_bytes(bytes(linker.complete().code))
                command = ['C:/Program Files/NVIDIA GPU Computing Toolkit/'
                           'CUDA/v13.3/bin/nvdisasm.exe', '-c', str(cubin)]
                process = subprocess.run(command, capture_output=True, text=True)
                disassembly = dict(command=command, exit_code=process.returncode,
                                   stdout=process.stdout, stderr=process.stderr)
                (folder / 'disassembly.json').write_text(json.dumps(disassembly, indent=2))
                process.check_returncode()
                ffma = len(re.findall(r'/\*[0-9a-f]+\*/\s*(?:@!?P\d+\s+)?FFMA\b', process.stdout))
                record = dict(source=source, fma=fma, ir=asset(ir_path),
                              original=asset(original), cubin=asset(cubin),
                              options=options, rendered_options=rendered,
                              native_ffma_sites=ffma,
                              exact_original_bytes=cubin.read_bytes() == original.read_bytes(),
                              info_log=linker.info_log, error_log=linker.error_log)
                receipt['records'].append(record)
                if fma and not record['exact_original_bytes']:
                    raise ValueError('Baseline relink changed native bytes')
        receipt['status'] = 'OFFLINE_DOUBLE_CONTRACTION_INTERVENTION_AUTHOR_COMPLETE'
    except Exception:
        receipt['status'] = 'FAILED_RETAINED'
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps([{key: row[key] for key in (
        'source', 'fma', 'native_ffma_sites', 'exact_original_bytes'
    )} for row in receipt['records']]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.capture, arguments.output)
