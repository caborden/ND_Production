"""CPU-only skim truth extraction and Plotly displays, in EDepSim xyz centimetres."""
import itertools
import json
from pathlib import Path
import h5py
import numpy as np

TARGETS = [(55, 12), (58, 5), (58, 15), (62, 17), (257, 26),
           (54, 16), (57, 6), (61, 14), (256, 17)]


def load_event(campaign, index, event_id):
    rows = json.loads((Path(campaign) / 'files.json').read_text())
    row, = [r for r in rows if r['index'] == index]
    with h5py.File(row['skim'], 'r') as data:
        records = [json.loads(r) for r in data['provenance/events'].asstr()[:]]
        record, = [r for r in records if r['output_event_id'] == event_id]
        run = json.loads(data['provenance'].attrs['run_json'])
        source = json.loads(data[f"provenance/sources/{record['source_index']}"].attrs['fingerprint_json'])
        result = dict(metadata=dict(index=index, skim=row['skim'], record=record,
                                    source=source, active_bounds_cm=run['active_bounds_cm']))
        for name in ('segments', 'trajectories', 'vertices'):
            if name in data:
                ds = data[name]
                ids = ds.fields('event_id')[:]
                result[name] = ds[:][ids == event_id]
    return result


def save_bundle(event, path):
    """Small portable truth file; never rewrites the simulation input."""
    with h5py.File(path, 'x') as data:
        data.attrs['metadata_json'] = json.dumps(event['metadata'])
        for name in ('segments', 'trajectories', 'vertices'):
            if name in event:
                data.create_dataset(name, data=event[name], compression='gzip')


def load_bundle(path):
    with h5py.File(path, 'r') as data:
        return dict(metadata=json.loads(data.attrs['metadata_json']),
                    **{name: data[name][:] for name in data})


def endpoints(event):
    rows = event['segments']
    start = np.column_stack([rows[f'{a}_start'] for a in 'xyz'])
    end = np.column_stack([rows[f'{a}_end'] for a in 'xyz'])
    return start, end


def endpoint_active(event):
    start, end = endpoints(event)
    bounds = np.asarray(event['metadata']['active_bounds_cm'])
    return (((start > bounds[:, 0]) & (start < bounds[:, 1])).all(axis=1)
            | ((end > bounds[:, 0]) & (end < bounds[:, 1])).all(axis=1))


def summary(event):
    start, end = endpoints(event)
    active = endpoint_active(event)
    projected = np.linalg.norm((end - start)[:, 1:3], axis=1)
    return dict(file=event['metadata']['index'],
                event=event['metadata']['record']['output_event_id'],
                original_event=event['metadata']['record']['source_event_id'],
                segments=len(start), endpoint_active=int(active.sum()),
                max_active_projected_segment_cm=float(projected[active].max()) if active.any() else None)


def figure(event, active_only=False, detector_view=True):
    import plotly.graph_objects as go
    rows = event['segments']
    start, end = endpoints(event)
    active = endpoint_active(event)
    bounds = np.asarray(event['metadata']['active_bounds_cm'])
    fig = go.Figure()
    species = np.abs(rows['pdg_id'])
    categories = [(13, 'muon', '#0072B2'), (11, 'electron/positron', '#E69F00'),
                  (2212, 'proton', '#009E73'), (22, 'photon', '#CC79A7'),
                  (2112, 'neutron', '#999999'), (None, 'other', '#D55E00')]
    for pdg, label, color in categories:
        mask = species == pdg if pdg is not None else ~np.isin(species, [13, 11, 2212, 22, 2112])
        if active_only:
            mask &= active
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        xyz = np.full((len(indices), 3, 3), np.nan)
        xyz[:, 0, :] = start[mask]
        xyz[:, 1, :] = end[mask]
        hover = []
        for i in indices:
            info = [f'{key}={rows[key][i]}' for key in
                    ('segment_id', 'traj_id', 'file_traj_id', 'pdg_id', 'dE', 't_start', 't_end')
                    if key in rows.dtype.names]
            info += [f'endpoint_active={bool(active[i])}',
                     f'length_cm={np.linalg.norm(end[i]-start[i]):.4g}',
                     f'projected_yz_cm={np.linalg.norm((end[i]-start[i])[1:]):.4g}']
            hover.extend(['<br>'.join(info)] * 3)
        xyz = xyz.reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines',
                                 line=dict(color=color, width=3), name=label,
                                 text=hover, hovertemplate='%{text}<extra></extra>'))
    edges = []
    for corner in itertools.product((0, 1), repeat=3):
        for axis in range(3):
            if corner[axis] == 0:
                other = list(corner)
                other[axis] = 1
                edges.extend([[bounds[a, corner[a]] for a in range(3)],
                              [bounds[a, other[a]] for a in range(3)], [np.nan] * 3])
    box = np.asarray(edges)
    fig.add_trace(go.Scatter3d(x=box[:, 0], y=box[:, 1], z=box[:, 2], mode='lines',
                             line=dict(color='black', width=5), name='active volume', hoverinfo='skip'))
    scene = dict(aspectmode='data', camera=dict(up=dict(x=0, y=1, z=0)))
    for a, name in enumerate('xyz'):
        scene[name + 'axis'] = dict(title=f'{name} [cm]')
        if detector_view:
            scene[name + 'axis']['range'] = [bounds[a, 0]-2, bounds[a, 1]+2]
    meta = event['metadata']
    fig.update_layout(scene=scene, height=750,
                      title=f"File {meta['index']}, event {meta['record']['output_event_id']} "
                            f"(original {meta['record']['source_event_id']})")
    return fig
