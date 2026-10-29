"""Truth preselection, deliberately separate from reconstructed-hit acceptance."""
import hashlib
import json

import numpy as np

from .geometry import clip_segments


def length_metrics(starts, ends):
    """Find both PCA endpoint chord length and sum of segments path length;
       metrics are passed to a configured length check requirement later on."""
    if not len(starts):
        return 0.0, 0.0
    points = np.concatenate([starts, ends])
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ axes[0]
    chord = np.linalg.norm(points[projection.argmax()] - points[projection.argmin()])
    path = np.linalg.norm(ends - starts, axis=1).sum()
    return float(chord), float(path)


def qualifying_translations(segments, shifts, bounds, min_length_cm):
    """Find FSD tile candidates for translation to the mock FSD Cube volume.
       One candidate per translation, regardless of the number of passing muons.

    Requirement for candidates is that both chord and deposited path length must strictly exceed the cut.
    This cut is a loose truth proxy: no hit-count, energy-loss, or reconstructed PCA cuts apply.
    """
    muons = segments[np.abs(segments['pdg_id'].astype(np.int64)) == 13]
    groups = [muons[muons['file_traj_id'] == tid]
              for tid in np.unique(muons['file_traj_id'])]
    candidates = []
    for index, shift in enumerate(shifts):
        passing = []
        for rows in groups:
            starts = np.column_stack([rows[f'{a}_start'] for a in 'xyz']) + shift
            ends = np.column_stack([rows[f'{a}_end'] for a in 'xyz']) + shift
            if not np.isfinite(starts).all() or not np.isfinite(ends).all():
                raise ValueError('Nonfinite muon segment coordinates')
            clipped_start, clipped_end = clip_segments(starts, ends, bounds)
            chord, path = length_metrics(clipped_start, clipped_end)
            if chord > min_length_cm and path > min_length_cm:
                passing.append(dict(source_file_traj_id=int(rows['file_traj_id'][0]),
                                    chord_cm=chord, path_cm=path))
        if passing:
            candidates.append(dict(translation_index=index, shift_cm=shift.tolist(),
                                   muons=passing))
    return candidates


def choose_translation(candidates, seed, source_sha256, event_id):
    """Among passing candidates, use seeded rng to choose one to translate to the mock FSD Cube volume.
       Function is uniform over qualifying translations, stable under reordering/packing."""
    if not candidates:
        return None
    key = json.dumps([seed, source_sha256, int(event_id)], separators=(',', ':'))
    digest = hashlib.sha256(key.encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))
    result = dict(candidates[int(rng.integers(len(candidates)))])
    result['qualifying_translation_count'] = len(candidates)
    return result
