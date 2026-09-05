"""Prepare a source-bound integer update motif for native inspection."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def asset(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source(body, form):
    update = {
        'predicated_move': '    @commit mov.b32 value, temporary;',
        'select': '    selp.u32 value, temporary, value, commit;',
    }[form]
    motif = '\n'.join(
        '    add.u32 temporary, value, 1;\n' + update
        for _ in range(body)
    )
    return '''.version 8.0
.target sm_89
.address_size 64
.visible .entry predicated_update_probe(
    .param .u64 endpoints, .param .u64 elapsed,
    .param .u32 iterations, .param .u32 active_warps,
    .param .u32 initial_value, .param .u32 predicate_lanes,
    .param .u32 endpoint_bound)
.maxntid 1024, 1, 1
{
    .reg .pred commit, inactive, again, bad, lane_nonzero;
    .reg .u32 tid, cta, lane, warp, n, count, population, threshold;
    .reg .u32 value, temporary, bound, thread_index, warp_index;
    .reg .u64 out, times, address, begin, end, ticks, byte_offset;
    ld.param.u64 out, [endpoints];
    ld.param.u64 times, [elapsed];
    ld.param.u32 n, [iterations];
    ld.param.u32 population, [active_warps];
    ld.param.u32 value, [initial_value];
    ld.param.u32 threshold, [predicate_lanes];
    ld.param.u32 bound, [endpoint_bound];
    mov.u32 tid, %tid.x;
    mov.u32 cta, %ctaid.x;
    and.b32 lane, tid, 31;
    shr.u32 warp, tid, 5;
    setp.ge.u32 inactive, warp, population;
    setp.lt.u32 commit, lane, threshold;
    bar.sync 0;
    @inactive bra JOIN;
    mov.u32 count, 0;
    mov.u64 begin, %clock64;
LOOP:
''' + motif + '''
    add.u32 count, count, 1;
    setp.lt.u32 again, count, n;
    @again bra LOOP;
    setp.gt.u32 bad, value, bound;
    @bad bra INVALID;
    mov.u64 end, %clock64;
    sub.u64 ticks, end, begin;
    mad.lo.u32 thread_index, cta, 1024, tid;
    mul.wide.u32 byte_offset, thread_index, 4;
    add.u64 address, out, byte_offset;
    st.global.u32 [address], value;
    setp.ne.u32 lane_nonzero, lane, 0;
    @lane_nonzero bra JOIN;
    mad.lo.u32 warp_index, cta, 32, warp;
    mul.wide.u32 byte_offset, warp_index, 8;
    add.u64 address, times, byte_offset;
    st.global.u64 [address], ticks;
    bra JOIN;
INVALID:
    trap;
JOIN:
    bar.sync 0;
    ret;
}
'''


def run(output):
    output.mkdir(exist_ok=False)
    toolkit = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin')
    receipt = dict(status='PREPARED_REQUIRES_NATIVE_REVIEW', source=asset(__file__),
                   cases=[], scope='Offline PTXAS/native inspection only; '
                   'no kernel launch. No promised native MOV or SEL form; '
                   'compiler rewrites remain observations.')
    for form in ('predicated_move', 'select'):
        for body in (33, 257):
            for optimization in (3, 0):
                folder = output / f'{form}_b{body}_O{optimization}'
                folder.mkdir()
                ptx = folder / 'kernel.ptx'
                ptx.write_text(source(body, form))
                cubin = folder / 'kernel.cubin'
                commands = []
                for command in (
                    [str(toolkit / 'ptxas.exe'), '-arch=sm_89',
                     f'-O{optimization}', '-v', str(ptx), '-o', str(cubin)],
                    [str(toolkit / 'nvdisasm.exe'), '-c', str(cubin)],
                ):
                    process = subprocess.run(command, capture_output=True, text=True)
                    commands.append(dict(command=command, exit_code=process.returncode,
                                         stdout=process.stdout, stderr=process.stderr))
                    (folder / 'commands.json').write_text(json.dumps(commands, indent=2))
                    process.check_returncode()
                (folder / 'kernel.sass').write_text(commands[-1]['stdout'])
                receipt['cases'].append(dict(
                    form=form, body=body, optimization=optimization,
                    ptx=asset(ptx), cubin=asset(cubin),
                    sass=asset(folder / 'kernel.sass'),
                    assembler=asset(toolkit / 'ptxas.exe'),
                    disassembler=asset(toolkit / 'nvdisasm.exe'),
                ))
                (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
                print(folder.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
