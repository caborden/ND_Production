"""Single-TPC geometry in edep-sim (x=drift, y=vertical, z=transverse) cm."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def fingerprint(path):
    """Stream the digest so hashing a large source does not duplicate it in RAM."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                sha256=digest.hexdigest())


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def validate_config(config):
    expected = {'schema_version', 'detector_config', 'min_length_cm', 'seed',
                'events_per_file', 'translation_grid'}
    if set(config) != expected or config['schema_version'] != 1:
        raise ValueError('Expected configuration schema 1 and exactly its documented keys')
    length = config['min_length_cm']
    if isinstance(length, bool) or not isinstance(length, (int, float)) or not np.isfinite(length) or length <= 0:
        raise ValueError('min_length_cm must be finite and positive')
    for key in ('seed', 'events_per_file'):
        if type(config[key]) is not int or config[key] < (1 if key == 'events_per_file' else 0):
            raise ValueError(f'Invalid {key}')
    if config['events_per_file'] > 10000:
        raise ValueError('events_per_file exceeds cosmic_sim.yaml max_events_per_file=10000')
    grid = config['translation_grid']
    if set(grid) != {'counts', 'steps_cm'}:
        raise ValueError('translation_grid needs counts and steps_cm')
    if len(grid['counts']) != 3 or any(type(n) is not int or n < 1 for n in grid['counts']):
        raise ValueError('counts must contain three positive integers')
    steps = np.asarray(grid['steps_cm'], dtype=float)
    if steps.shape != (3,) or not np.isfinite(steps).all() or (steps <= 0).any():
        raise ValueError('steps_cm must contain three finite positive lengths')
    return config


def cube_geometry(larnd_repo, config_name):
    """Read YAML without importing larnd-sim/CUDA; refuse unsupported geometries."""
    root = Path(larnd_repo).resolve() / 'larndsim'
    registry = root / 'config/config.yaml'
    cfg = yaml.safe_load(registry.read_text())[config_name]
    layout_path = root / 'pixel_layouts' / cfg['PIXEL_LAYOUT']
    properties_path = root / 'detector_properties' / cfg['DET_PROPERTIES']
    layout = yaml.safe_load(layout_path.read_text())
    properties = yaml.safe_load(properties_path.read_text())
    positions = layout['tile_positions']
    offsets = np.asarray(properties['tpc_offsets'], dtype=float)
    if len(positions) != 1 or offsets.shape != (1, 3):
        raise ValueError('This skim currently supports one tile and one TPC only')
    tile = next(iter(positions))
    center = np.asarray(positions[tile], dtype=float) / 10 + offsets[0]
    direction = layout['tile_orientations'][tile][0]
    drift = float(layout['drift_length']) / 10
    if direction not in (-1, 1) or drift <= 0:
        raise ValueError('Invalid drift length or cathode direction')
    pixels = np.asarray(list(layout['chip_channel_to_position'].values()))
    if np.any(pixels.min(axis=0) != 0):
        raise ValueError('Expected zero-based pixel coordinates')
    half = (pixels.max(axis=0) + 1) * float(layout['pixel_pitch']) / 20
    bounds = np.array([sorted([center[0], center[0] + direction * drift]),
                       [center[1] - half[1], center[1] + half[1]],
                       [center[2] - half[0], center[2] + half[0]]])
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Invalid active-volume bounds')
    return bounds, [fingerprint(p) for p in (registry, layout_path, properties_path)]


def translations(config):
    """Construct the configured translation vectors to move simulated event coordinates in each FSD tile volume to the mock FSD Cube tile volume."""
    grid = config['translation_grid']
    return np.array(list(np.ndindex(*grid['counts'])), dtype=float) * grid['steps_cm']


def clip_segments(starts, ends, bounds):
    """Clip edep segement endpoints using slab intersection to the borders of the FSD Cube volume."""
    delta = ends - starts
    enter, leave = np.zeros(len(starts)), np.ones(len(starts))
    valid = np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
    for axis, (lower, upper) in enumerate(bounds):
        moving = delta[:, axis] != 0
        valid &= moving | ((starts[:, axis] > lower) & (starts[:, axis] < upper))
        low = np.full(len(starts), -np.inf)
        high = np.full(len(starts), np.inf)
        low[moving] = (lower - starts[moving, axis]) / delta[moving, axis]
        high[moving] = (upper - starts[moving, axis]) / delta[moving, axis]
        enter = np.maximum(enter, np.minimum(low, high))
        leave = np.minimum(leave, np.maximum(low, high))
    keep = valid & (leave > enter)
    return starts[keep] + enter[keep, None] * delta[keep], starts[keep] + leave[keep, None] * delta[keep]
