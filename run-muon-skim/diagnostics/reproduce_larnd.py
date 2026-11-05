#!/usr/bin/env python3
"""Run one campaign input unchanged; observe only the signals allocation."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import weakref
import json
import math
import os
from pathlib import Path
import sys


def signals_lines(script):
    """Fail closed if the installed simulator no longer has the expected call."""
    matches = []
    for node in ast.walk(ast.parse(Path(script).read_text())):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'signals' for t in node.targets)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == 'cp'
                and node.value.func.attr == 'zeros'):
            matches.append((node.value.lineno, node.value.end_lineno))
    if len(matches) != 1:
        raise ValueError(f'Expected one signals = cp.zeros call; found {matches}')
    return matches[0]


def memory(cp):
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return dict(pool_used_bytes=pool.used_bytes(), pool_total_bytes=pool.total_bytes(),
                pool_cached_bytes=pool.free_bytes(), pool_limit_bytes=pool.get_limit(),
                cuda_free_bytes=free, cuda_total_bytes=total)


def instrument(cp, script, emit):
    """Instrument one AST call, preserving allocation arguments and array lifetimes."""
    first, last = signals_lines(script)
    tree = ast.parse(Path(script).read_text())
    context = ast.parse("dict(ievd=int(ievd), i_mod=int(i_mod), "
                        "start_pix=int(start_pix), stop_pix=int(stop_pix), "
                        "max_pixels=int(max_pixels[0]), max_signal_time=float(max_signal_time), "
                        "time_sampling=float(detector.TIME_SAMPLING))", mode='eval').body
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.lineno == first and node.end_lineno == last:
            if (isinstance(node.func, ast.Attribute) and node.func.attr == 'zeros'):
                node.func = ast.Name(id='__diagnostic_signals_zeros', ctx=ast.Load())
                node.keywords.append(ast.keyword(arg='_diagnostic_context', value=context))
                break
    ast.fix_missing_locations(tree)
    code = compile(tree, str(Path(script).resolve()), 'exec')
    original = cp.zeros
    previous = None

    def zeros(shape, *args, _diagnostic_context, **kwargs):
        nonlocal previous
        # A weak reference observes the previous signals tensor without keeping
        # it alive. Do not inspect frame.f_locals: its cache can retain arrays.
        old = previous() if previous is not None else None
        old_bytes = int(old.nbytes) if old is not None else 0
        del old
        dtype = cp.dtype(kwargs.get('dtype', args[0] if args else float))
        context = dict(_diagnostic_context, shape=[int(n) for n in shape],
                       dtype=str(dtype), signals_ticks=int(shape[2]),
                       requested_bytes=math.prod(int(n) for n in shape) * dtype.itemsize,
                       previous_signals_bytes=old_bytes)
        emit(dict(phase='before', **context, **memory(cp)))
        try:
            result = original(shape, *args, **kwargs)
        except Exception as error:
            emit(dict(phase='error', **context, **memory(cp),
                      error_type=type(error).__name__, error=str(error)))
            raise
        previous = weakref.ref(result)
        emit(dict(phase='allocated', **context, **memory(cp)))
        return result
    return code, zeros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--index', type=int, default=62)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign.resolve()
    config_path = root / 'campaign.json'
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if digest != (root / 'campaign.sha256').read_text().strip():
        raise ValueError('Campaign configuration checksum mismatch')
    config = json.loads(config_path.read_text())
    spec = importlib.util.spec_from_file_location('frozen_campaign', root / 'campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    campaign.guard(config)
    rows = json.loads((root / 'files.json').read_text())
    row, = [r for r in rows if r['index'] == args.index]
    script = Path(config['envs']['larnd']).resolve(strict=True)
    signals_lines(script)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(script), 'fsd_cube', '--input_filename', row['skim'],
               '--output_filename', str(output / 'simulation.LARNDSIM.hdf5'),
               '--rand_seed', str(args.index + 1), '--compression', 'None']
    import cupy as cp
    metadata = dict(command=command, cwd=config['work'], input=row,
                    campaign_sha256=digest, python=sys.version, cupy=cp.__version__,
                    simulator_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    repos=config['repos'], device_id=cp.cuda.runtime.getDevice(),
                    device_properties=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice()),
                    environment={k: v for k, v in os.environ.items()
                                 if k.startswith(('CUDA', 'CUPY', 'SLURM'))})
    (output / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    old_argv, old_cwd = sys.argv, Path.cwd()
    with (output / 'allocations.jsonl').open('x', buffering=1) as stream:
        def emit(record):
            record['utc'] = datetime.now(timezone.utc).isoformat()
            stream.write(json.dumps(record) + '\n')
            print('SIGNALS_ALLOCATION ' + json.dumps(record), flush=True)
        code, zeros = instrument(cp, script, emit)
        sys.argv = command
        os.chdir(config['work'])
        try:
            exec(code, dict(__name__='__main__', __file__=str(script),
                            __package__=None, __diagnostic_signals_zeros=zeros))
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
    (output / 'SIMULATION_RETURNED_SUCCESSFULLY').write_text('Diagnostic only; not a campaign receipt.\n')


if __name__ == '__main__':
    main()
