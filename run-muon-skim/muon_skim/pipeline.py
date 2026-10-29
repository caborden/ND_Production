"""Discover sources, scan events, pack selected events, and record a run manifest."""
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import platform

import h5py
import numpy as np
import yaml

from .geometry import cube_geometry, fingerprint, translations, validate_config
from .io import OutputFileWriter, events, read_tables
from .selection import choose_translation, qualifying_translations


def discover(inputs=(), patterns=(), input_list=None):
    """Resolve real paths; reject duplicates and any unmatched explicit request."""
    paths = []
    requested = list(inputs)
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise ValueError(f'No inputs match {pattern}')
        requested.extend(matches)
    if input_list:
        listing = Path(input_list).resolve()
        for line in listing.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                path = Path(line.strip()).expanduser()
                requested.append(path if path.is_absolute() else listing.parent / path)
    for value in requested:
        path = Path(value).expanduser().resolve()
        matches = sorted(path.rglob('*.hdf5')) if path.is_dir() else [path]
        if not matches:
            raise ValueError(f'No HDF5 inputs in {path}')
        for match in matches:
            match = match.resolve()
            if not match.is_file():
                raise FileNotFoundError(match)
            if match in paths:
                raise ValueError(f'Duplicate input: {match}')
            paths.append(match)
    if not paths:
        raise ValueError('No input files requested')
    return sorted(paths)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def run(inputs, output_dir, config, larnd_repo, prefix='muons', config_source=None):
    """Run once into a new directory. On failure retain completed output files and audit.

    Inputs must be unoverlaid convert2h5 cosmic events with event-relative times.
    Time values and particle momenta are copied without modification.
    """
    config = validate_config(config)
    paths = discover(inputs)
    if not prefix or Path(prefix).name != prefix or prefix in {'.', '..'}:
        raise ValueError('prefix must be a plain filename component')
    bounds, geometry_sources = cube_geometry(larnd_repo, config['detector_config'])
    shifts = translations(config)
    metadata = dict(config=config, geometry_sources=geometry_sources,
                    active_bounds_cm=bounds.tolist(), translations_cm=shifts.tolist(),
                    coordinate_system='edep-sim xyz cm; x=drift, y=vertical',
                    selection='abs(pdg_id)==13; clipped PCA chord and path strictly > min_length_cm',
                    placement='uniform qualifying translations; one copy per source event',
                    rng='PCG64 seeded by SHA256([seed, source_sha256, source_event_id])',
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  h5py=h5py.__version__, pyyaml=yaml.__version__),
                    code=[fingerprint(p) for p in sorted(Path(__file__).parent.glob('*.py'))])
    if config_source:
        metadata['config_source'] = fingerprint(config_source)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, status='incomplete', metadata=metadata,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    sources=[], outputs=[], scanned_events=0, selected_events=0)
    manifest_path = directory / 'manifest.json'
    writer = None
    write_json(manifest_path, manifest)
    try:
        for source_index, path in enumerate(paths):
            info = fingerprint(path)
            manifest['sources'].append(info)
            with h5py.File(path, 'r') as source:
                tables = read_tables(source)
                schema = {name: rows.dtype for name, rows in tables.items()}
                for event_id, event in events(tables):
                    manifest['scanned_events'] += 1
                    candidates = qualifying_translations(event['segments'], shifts, bounds, config['min_length_cm'])
                    choice = choose_translation(candidates, config['seed'], info['sha256'], event_id)
                    if choice is None:
                        continue
                    if writer is not None and (writer.count >= config['events_per_file'] or writer.schema != schema):
                        manifest['outputs'].append(writer.finish())
                        writer = None
                        write_json(manifest_path, manifest)
                    if writer is None:
                        target = directory / f'{prefix}.{len(manifest["outputs"]):07d}.EDEPSIM.hdf5'
                        writer = OutputFileWriter(target, schema, metadata)
                    writer.append(event, source, source_index, info, event_id, choice)
                    manifest['selected_events'] += 1
            stat = path.stat()
            if stat.st_size != info['size_bytes'] or stat.st_mtime_ns != info['mtime_ns']:
                raise ValueError(f'Source changed during scan: {path}')
            write_json(manifest_path, manifest)
            print(f'Scanned {path.name}: {manifest["selected_events"]}/{manifest["scanned_events"]} events selected', flush=True)
        if writer is not None:
            manifest['outputs'].append(writer.finish())
            writer = None
        # Empty selections intentionally yield an empty list, not unusable empty MC.
        (directory / 'larnd_inputs.txt').write_text(''.join(f'{row["path"]}\n' for row in manifest['outputs']))
        manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
    except Exception as exc:
        if writer is not None:
            writer.close_incomplete()
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(manifest_path, manifest)
        raise
    return manifest
