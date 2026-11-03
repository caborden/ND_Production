#!/usr/bin/env python3
"""Prepare and submit a dependency-gated NERSC muon campaign from a completed pilot.

Preparation is read-only with respect to the pilot and existing checkouts. Jobs
use frozen skim/selection sources, existing pilot environments, and guarded
larnd-sim/Flow checkouts. No scheduler calls occur until `submit` is requested.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import h5py
import numpy as np
import yaml

FLOW_COMMIT = '916fb2e1f023f95206632f4a839ab923c6561751'
TRACK_FILES = ('track_selection_fsd.py', 'batch_track_selection.py',
               'track_output_hdf5.py', 'merge_track_outputs.py', 'efield.py')
WORKFLOWS = (
    'charge/charge_event_building_mc.yaml',
    'charge/charge_event_reconstruction_mc.yaml',
    'combined/combined_reconstruction_mc.yaml',
    'charge/prompt_calibration_mc.yaml',
    'charge/filtered_calibration_mc.yaml',
)
SOURCE_DIR = '/pscratch/sd/d/dunepro/cuddandr/output/FSD_CosmicRun4/run-convert2h5/FSD_CosmicRun4.convert2h5/EDEPSIM_H5'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_state(repo):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    return dict(commit=git('rev-parse', 'HEAD').decode().strip(),
                diff_sha256=hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest())


def groups(records, workers, worker):
    require(workers > 0 and 0 <= worker < workers, 'Invalid worker index')
    return records[worker::workers]


def h5_check(path, required):
    with h5py.File(path, 'r') as data:
        for name in required:
            require(name in data and isinstance(data[name], h5py.Dataset),
                    f'{path}: missing dataset {name}')
            if len(data[name]):
                data[name][0:1]  # Exercise a data read, not just the header.


def check_flow(path):
    h5_check(path, ['charge/events/data', 'charge/calib_prompt_hits/data',
                   'charge/calib_filtered_hits/data', 'charge/ext_trigs/data',
                   'combined/t0/data', 'mc_truth/segments/data',
                   'charge/events/ref/charge/calib_prompt_hits/ref',
                   'charge/events/ref/charge/calib_prompt_hits/ref_region',
                   'charge/events/ref/charge/ext_trigs/ref',
                   'charge/events/ref/charge/ext_trigs/ref_region'])
    with h5py.File(path, 'r') as data:
        bounds = data['geometry_info'].attrs['lar_detector_bounds']
        require(np.isfinite(bounds).all(), f'{path}: nonfinite geometry')


def check_tracks(path, code, expected_source=None):
    with h5py.File(path, 'r') as data:
        require(data.attrs.get('status') == 'complete' and
                data.attrs.get('schema_version') == '2.0', f'Incomplete tracks: {path}')
        require(list(data['sources']) == ['0'], f'Expected per-input tracks: {path}')
        require(data['sources/0'].attrs.get('status') == 'complete', f'Incomplete source: {path}')
        metadata = data['sources/0/metadata'].attrs
        require(metadata['min_length_cm'] == 29 and not metadata['is2x2'],
                f'{path}: expected FSD selection with strict >29 cm')
        if expected_source is not None:
            require(Path(metadata['input_file']).resolve() == Path(expected_source).resolve(),
                    f'{path}: wrong FLOW source')
        for name in ('track_selection_fsd.py', 'efield.py', 'track_output_hdf5.py'):
            require(metadata[name + '_sha256'] == sha(Path(code) / name),
                    f'{path}: selection code differs from pilot ({name})')
        lengths = data['tracks/data']['length_cm']
        require(np.isfinite(lengths).all() and (lengths > 29).all(), f'{path}: invalid lengths')
        return len(lengths)


def prepare(args):
    work = Path(args.work_root).resolve()
    jobs = work / 'muon-skim-jobs'
    output_base = Path(os.environ.get('ND_PRODUCTION_OUTDIR_BASE',
                                     str(Path(os.environ['SCRATCH']) / 'fsd-muon-production/output')))
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.name), 'Invalid campaign name')
    require(args.additional_sources > 0 and 1 <= args.gpu_workers <= 10 and
            1 <= args.flow_workers <= 5, 'Invalid file or worker count')
    require(args.cpu_account and args.gpu_account and
            not args.cpu_account.startswith('YOUR_') and not args.gpu_account.startswith('YOUR_'),
            'Set CPU_ACCOUNT and GPU_ACCOUNT, or provide --cpu-account and --gpu-account')
    root = jobs / args.name
    outputs = Path(os.environ['SCRATCH']) / 'fsd-muon-production/campaigns' / args.name
    require(not root.exists() and not outputs.exists(), 'Campaign already exists; choose a new name')
    pilot_dir = output_base / 'run-muon-skim/FSDCubeMuons_pilot_v1/EDEPSIM_H5/0000000/FSDCubeMuons_pilot_v1.0000000'
    pilot = read(pilot_dir / 'manifest.json')
    require(pilot['status'] == 'complete', 'Pilot skim is incomplete')
    require(pilot['metadata']['config']['min_length_cm'] == 29, 'Pilot skim cut is not 29 cm')
    old_sources = {Path(row['path']).resolve() for row in pilot['sources']}
    candidates = sorted({p.resolve() for p in Path(args.source_dir).rglob('*.EDEPSIM.hdf5')})
    sources = [p for p in candidates if p not in old_sources][:args.additional_sources]
    require(len(sources) == args.additional_sources, 'Not enough new source files')
    track_code = jobs / 'track-selection-29cm-v1/code'
    track_manifest_path = Path(os.environ['SCRATCH']) / 'fsd-muon-production/track-selection/FSDCubeMuons_pilot_v1.min29cm/parts/batch_manifest.json'
    track_manifest = read(track_manifest_path)
    require(track_manifest['status'] == 'complete' and track_manifest['min_length_cm'] == 29,
            'Pilot track batch is incomplete or uses another length cut')
    pilot_parts = []
    pilot_tracks = 0
    for row in track_manifest['files']:
        require(row['status'] == 'complete', 'Pilot track file incomplete')
        pilot_tracks += check_tracks(row['output'], track_code, row['input'])
        pilot_parts.append(identity(row['output']))
    require(pilot_parts, 'No pilot track outputs')
    require(len(pilot_parts) == len(pilot['outputs']), 'Pilot skim/track file count mismatch')

    production = work / 'ND_Production-muon-skim'
    larnd = work / 'larnd-sim'
    flow = work / 'flow-fsdcube-916fb2e1/ndlar_flow'
    require(git_state(flow)['commit'] == FLOW_COMMIT, 'Flow checkout is not the reviewed pilot revision')
    saved_larnd_commit = jobs / 'larnd-sim.commit.txt'
    require(saved_larnd_commit.read_text().strip() == git_state(larnd)['commit'],
            'larnd-sim commit differs from saved pilot revision')
    for row in pilot['metadata']['geometry_sources']:
        require(sha(row['path']) == row['sha256'], 'Geometry/config registry changed since pilot')
    for row in pilot['metadata']['code']:
        path = production / 'run-muon-skim/muon_skim' / Path(row['path']).name
        require(sha(path) == row['sha256'], f'Skim code changed since pilot: {path}')
    pedestal_config = flow / 'yamls/fsdcube_flow/reco/charge/CalibHitBuilderMC.yaml'
    pedestal = yaml.safe_load(pedestal_config.read_text())['params']['pedestal_file']
    registry = yaml.safe_load((larnd / 'larndsim/config/config.yaml').read_text())['fsd_cube']
    require(registry['LIGHT_SIMULATED'] is False, 'Expected charge-only fsd_cube')
    response = Path(registry['RESPONSE'])
    if not response.is_absolute():
        response = larnd / 'larndsim/bin' / response
    external = [identity(pedestal), identity(response)]
    for workflow in WORKFLOWS:
        require((flow / 'yamls/fsdcube_flow/workflows' / workflow).is_file(), f'Missing {workflow}')
    envs = dict(skim=str(work / 'skim.venv/bin/python'),
                larnd=str(work / 'larnd.venv/bin/simulate_pixels.py'),
                flow=str(work / 'flow-fsdcube-916fb2e1/flow.venv/bin/h5flow'),
                analysis=args.analysis_env)
    for key in ('skim', 'larnd', 'flow'):
        require(os.access(envs[key], os.X_OK), f'Missing executable: {envs[key]}')
    require(Path(envs['analysis'], 'bin/python').is_file(), 'Missing analysis environment')

    root.mkdir(parents=True)
    outputs.mkdir(parents=True)
    (outputs / 'locks').mkdir()
    (root / 'logs').mkdir()
    shutil.copytree(production / 'run-muon-skim/muon_skim', root / 'code/muon_skim',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in TRACK_FILES:
        shutil.copy2(track_code / name, root / 'code' / name)
    shutil.copy2(__file__, root / 'campaign.py')
    write(root / 'skim-config.json', pilot['metadata']['config'])
    config = dict(name=args.name, root=str(root), outputs=str(outputs), envs=envs,
                  work=str(work), larnd=str(larnd), flow=str(flow),
                  sources=[identity(p) for p in sources],
                  pilot_source_hashes=[row['sha256'] for row in pilot['sources']],
                  pilot_parts=pilot_parts, pilot_tracks=pilot_tracks,
                  seed_offset=len(pilot['outputs']), gpu_workers=args.gpu_workers,
                  flow_workers=args.flow_workers, cpu_account=args.cpu_account,
                  gpu_account=args.gpu_account, external=external,
                  repos={str(p): git_state(p) for p in (larnd, flow)},
                  hashes={str(p): sha(p) for p in sorted((root / 'code').rglob('*.py'))})
    config['hashes'][str(root / 'skim-config.json')] = sha(root / 'skim-config.json')
    config['hashes'][str(root / 'campaign.py')] = sha(root / 'campaign.py')
    (root / 'stage.sh').write_text(stage_script(config))
    config['hashes'][str(root / 'stage.sh')] = sha(root / 'stage.sh')
    write(root / 'campaign.json', config)
    (root / 'campaign.sha256').write_text(sha(root / 'campaign.json') + '\n')
    print(json.dumps(dict(campaign=str(root / 'campaign.json'),
                          new_sources=[str(p) for p in sources], pilot_tracks=pilot_tracks,
                          expected_total_tracks=pilot_tracks * (1 + len(sources) / len(old_sources))), indent=2))
    print('Prepared only; submit with:')
    print(shlex.join([envs['skim'], str(root / 'campaign.py'), 'submit', str(root / 'campaign.json')]))


def stage_script(c):
    q = shlex.quote
    return f'''#!/bin/bash
set -eo pipefail
stage=$1
if [[ "$stage" == tracks ]]; then
    module load conda
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate {q(c['envs']['analysis'])}
    controller=$(command -v python)
else
    module load python/3.11
    controller={q(c['envs']['skim'])}
    if [[ "$stage" == larnd ]]; then
        module load cudatoolkit/12.4
        source {q(str(Path(c['work']) / 'larnd.venv/bin/activate'))}
    fi
fi
export H5FLOW_NOMPI=1 HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
env -u SLURM_TRES_PER_TASK srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \\
    "$controller" {q(str(Path(c['root']) / 'campaign.py'))} run \\
    {q(str(Path(c['root']) / 'campaign.json'))} "$stage" "${{SLURM_ARRAY_TASK_ID:-0}}"
'''


def guard(c):
    for path, digest in c['hashes'].items():
        require(sha(path) == digest, f'Frozen campaign code/config changed: {path}')
    for path, state in c['repos'].items():
        require(git_state(path) == state, f'Checkout changed while campaign active: {path}')
    for entry in c['external'] + c['sources'] + c['pilot_parts']:
        require(identity(entry['path']) == entry, f'Input/calibration file changed: {entry["path"]}')


def execute(command, log, cwd):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join([str(v) for v in command]), flush=True)
    with Path(log).open('a') as stream:
        stream.write('\n' + datetime.now(timezone.utc).isoformat() + '\n')
        stream.flush()
        subprocess.run([str(v) for v in command], cwd=cwd, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def archive_partial(path):
    path = Path(path)
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path.rename(path.with_name(path.name + '.failed-' + stamp))


def receipt_path(output):
    return Path(str(output) + '.receipt.json')


def receipt_valid(source, output, check):
    require(read(receipt_path(output)) == dict(input=identity(source), output=identity(output)),
            f'Missing or mismatched completion receipt: {output}')
    check(output)


def run_file(source, output, command, log, cwd, check):
    output = Path(output)
    if output.exists():
        if receipt_path(output).exists():
            receipt_valid(source, output, check)
            return
        # A crash between publishing the HDF5 and its receipt is not proof of
        # success. Preserve that file and regenerate it on an explicit retry.
        archive_partial(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + '.partial')
    archive_partial(partial)
    execute(command(partial), log, cwd)
    check(partial)
    os.link(partial, output)
    partial.unlink()
    write(receipt_path(output), dict(input=identity(source), output=identity(output)))


def skim_stage(c):
    root, out = Path(c['root']), Path(c['outputs'])
    rows, seen = [], set(c['pilot_source_hashes'])
    for index, source in enumerate(c['sources']):
        directory = out / 'skim' / f'{index:07d}'
        manifest_path = directory / 'manifest.json'
        complete = manifest_path.exists() and read(manifest_path).get('status') == 'complete'
        if not complete:
            archive_partial(directory)
            execute([c['envs']['skim'], '-m', 'muon_skim', source['path'],
                     '--config', root / 'skim-config.json', '--larnd-repo', c['larnd'],
                     '--output-dir', directory, '--prefix', f'source{index:07d}'],
                    root / 'logs' / f'skim-{index:07d}.log', root / 'code')
        manifest = read(manifest_path)
        require(manifest['status'] == 'complete', f'Skim failed: {directory}')
        require(manifest['metadata']['config'] == read(root / 'skim-config.json'), 'Skim config mismatch')
        require(len(manifest['sources']) == 1 and
                Path(manifest['sources'][0]['path']).resolve() == Path(source['path']), 'Skim source mismatch')
        digest = manifest['sources'][0]['sha256']
        require(digest not in seen, 'Duplicate original source content, including pilot')
        seen.add(digest)
        count = 0
        for record in manifest['outputs']:
            path = record['path']
            h5_check(path, ['segments', 'trajectories', 'vertices', 'provenance/events'])
            with h5py.File(path, 'r') as data:
                require(data.attrs.get('status') == 'complete', 'Incomplete skim HDF5')
                require(len(data['provenance/events']) == record['events'], 'Skim event count mismatch')
            count += record['events']
            i = len(rows) + c['seed_offset']
            rows.append(dict(index=i, skim=path,
                             larnd=str(out / 'larnd' / f'{i:07d}.LARNDSIM.hdf5'),
                             flow=str(out / 'flow' / f'{i:07d}.FLOW.hdf5')))
        require(count == manifest['selected_events'], 'Skim manifest totals mismatch')
    require(rows, 'No selected events; refusing to launch an empty downstream campaign')
    write(root / 'files.json', rows)


def larnd_check(path):
    h5_check(path, ['packets', 'mc_packets_assn', 'segments', 'trajectories', 'vertices'])
    with h5py.File(path, 'r') as data:
        require(len(data['packets']) == len(data['mc_packets_assn']),
                f'{path}: packet/truth-association length mismatch')


def larnd_stage(c, worker):
    root = Path(c['root'])
    for row in groups(read(root / 'files.json'), c['gpu_workers'], worker):
        run_file(row['skim'], row['larnd'],
                 lambda partial: [c['envs']['larnd'], 'fsd_cube', '--input_filename', row['skim'],
                                  '--output_filename', partial, '--rand_seed', row['index'] + 1,
                                  '--compression', 'None'],
                 root / 'logs' / f'larnd-{row["index"]:07d}.log', c['work'], larnd_check)


def flow_stage(c, worker):
    root = Path(c['root'])
    rows = read(root / 'files.json')
    # Global gate: every GPU output must be complete before any Flow worker starts.
    for row in rows:
        receipt_valid(row['skim'], row['larnd'], larnd_check)
    for row in groups(rows, c['flow_workers'], worker):
        run_file(row['larnd'], row['flow'],
                 lambda partial: [c['envs']['flow'], '--nompi', '-i', row['larnd'], '-o', partial,
                                  '-c', *[str(Path(c['flow']) / 'yamls/fsdcube_flow/workflows' / p)
                                          for p in WORKFLOWS]],
                 root / 'logs' / f'flow-{row["index"]:07d}.log', c['flow'], check_flow)


def tracks_stage(c):
    root, out = Path(c['root']), Path(c['outputs'])
    rows = read(root / 'files.json')
    for row in rows:
        receipt_valid(row['larnd'], row['flow'], check_flow)
    sys.path.insert(0, str(root / 'code'))
    from batch_track_selection import run_batch
    from merge_track_outputs import merge_outputs
    parts = run_batch([r['flow'] for r in rows], out / 'tracks/parts',
                      resume=True, min_length_cm=29)
    new_tracks = sum(check_tracks(p, root / 'code', r['flow']) for p, r in zip(parts, rows))
    require(new_tracks > 0, 'No new accepted tracks; inspect selection logs')
    pilot_parts = [p['path'] for p in c['pilot_parts']]
    for path in pilot_parts:
        check_tracks(path, root / 'code')
    final = out / 'tracks' / f'{c["name"]}.tracks.hdf5'
    merge_outputs([*pilot_parts, *parts], final, resume=True)
    with h5py.File(final, 'r') as data:
        require(data.attrs['status'] == 'complete', 'Final merge is incomplete')
        total = len(data['tracks/data'])
        require(total == c['pilot_tracks'] + new_tracks, 'Final track count mismatch')
        require(len(data['sources']) == len(pilot_parts) + len(parts), 'Final source count mismatch')
    write(root / 'summary.json', dict(status='complete', new_original_files=len(c['sources']),
                                     new_flow_files=len(rows), pilot_tracks=c['pilot_tracks'],
                                     new_tracks=new_tracks, total_tracks=total, output=str(final)))
    print((root / 'summary.json').read_text(), flush=True)


def lock_path(c, name):
    # The campaign root is on CFS, whose compute-node DVS access does not
    # support flock. All submitters/workers must share locks on scratch.
    directory = Path(c['outputs']) / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def run_stage(c, stage, worker):
    require(stage not in ('skim', 'tracks') or worker == 0, 'Single-worker stage requires index zero')
    lock = lock_path(c, f'.{stage}-{worker}.lock')
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(c)
        if stage == 'skim':
            skim_stage(c)
        elif stage == 'larnd':
            larnd_stage(c, worker)
        elif stage == 'flow':
            flow_stage(c, worker)
        else:
            tracks_stage(c)
        guard(c)


def submit(c, retry=False):
    root = Path(c['root'])
    with lock_path(c, '.submit.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = root / 'submission.json'
        ledger = read(ledger_path) if ledger_path.exists() else dict(attempts=[])
        if ledger['attempts']:
            require(retry, 'Already submitted; inspect jobs, then use --retry after they terminate')
            ids = [job['id'] for attempt in ledger['attempts'] for job in attempt['jobs']]
            if ids:
                active = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(),
                                                  '--format=%i'], text=True)
                active_ids = {line.strip().split('_')[0] for line in active.splitlines()}
                require(not active_ids.intersection(ids), 'Earlier campaign jobs are still queued/running')
        guard(c)
        attempt = dict(jobs=[], status='submitting')
        ledger['attempts'].append(attempt)
        write(ledger_path, ledger)
        try:
            previous = None
            for stage, workers, hours in [('skim', 1, 2), ('larnd', c['gpu_workers'], 6),
                                           ('flow', c['flow_workers'], 3), ('tracks', 1, 3)]:
                gpu = stage == 'larnd'
                command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--nodes=1',
                           '--ntasks=1', '--qos=shared', '--constraint=' + ('gpu' if gpu else 'cpu'),
                           '--cpus-per-task=' + ('32' if gpu else '2'), f'--time={hours}:00:00',
                           '--account=' + c['gpu_account' if gpu else 'cpu_account'],
                           '--job-name=' + c['name'] + '-' + stage,
                           '--chdir=' + str(root), '--output=' + str(root / 'logs' / (stage + '-%A_%a.out'))]
                if gpu:
                    command.append('--gpus-per-task=1')
                if stage in ('larnd', 'flow'):
                    command.append(f'--array=0-{workers-1}%{workers}')
                if previous:
                    command.append('--dependency=afterok:' + previous)
                else:
                    command.append('--hold')  # Start only after the full dependency chain exists.
                result = subprocess.check_output(command + [str(root / 'stage.sh'), stage], text=True)
                job_id = result.strip().split(';')[0]
                require(job_id.isdecimal(), f'Unexpected sbatch response: {result}')
                attempt['jobs'].append(dict(stage=stage, id=job_id))
                write(ledger_path, ledger)
                previous = job_id
            subprocess.run(['scontrol', 'release', attempt['jobs'][0]['id']], check=True)
            attempt['status'] = 'submitted'
        except BaseException:
            attempt['status'] = 'submission_failed'
            for job in attempt['jobs']:
                subprocess.run(['scancel', job['id']], check=False)
            raise
        finally:
            write(ledger_path, ledger)
        print(json.dumps(attempt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--name', required=True)
    prep.add_argument('--work-root', default='/global/cfs/cdirs/dune/users/cborden')
    prep.add_argument('--source-dir', default=SOURCE_DIR)
    prep.add_argument('--additional-sources', type=int, default=5)
    prep.add_argument('--gpu-workers', type=int, default=10)
    prep.add_argument('--flow-workers', type=int, default=5)
    prep.add_argument('--cpu-account', default=os.environ.get('CPU_ACCOUNT'))
    prep.add_argument('--gpu-account', default=os.environ.get('GPU_ACCOUNT'))
    prep.add_argument('--analysis-env', default='/global/common/software/dune/efield2_rcmanduj/')
    sub = commands.add_parser('submit')
    sub.add_argument('config')
    sub.add_argument('--retry', action='store_true')
    run = commands.add_parser('run')
    run.add_argument('config')
    run.add_argument('stage', choices=['skim', 'larnd', 'flow', 'tracks'])
    run.add_argument('worker', type=int, nargs='?', default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        config_path = Path(args.config)
        require(sha(config_path) == config_path.with_name('campaign.sha256').read_text().strip(),
                'Campaign configuration changed; prepare a new campaign instead')
        if args.command == 'submit':
            submit(read(config_path), args.retry)
        else:
            run_stage(read(config_path), args.stage, args.worker)


if __name__ == '__main__':
    main()
