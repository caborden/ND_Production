"""Schema-aware event I/O, explicit truth mappings, and exclusive publication."""
import json
import os
from pathlib import Path

import h5py
import numpy as np


REQUIRED = {
    'segments': {'event_id', 'vertex_id', 'segment_id', 'traj_id', 'file_traj_id',
                 'pdg_id', 'x', 'y', 'z', 'x_start', 'y_start', 'z_start',
                 'x_end', 'y_end', 'z_end'},
    'trajectories': {'event_id', 'vertex_id', 'traj_id', 'file_traj_id', 'parent_id',
                     'xyz_start', 'xyz_end'},
    'vertices': {'event_id', 'vertex_id', 'x_vert', 'y_vert', 'z_vert'},
}
OPTIONAL = {'mc_stack', 'mc_hdr'}
POSITIONS = {
    'segments': [('x', 'y', 'z'), ('x_start', 'y_start', 'z_start'), ('x_end', 'y_end', 'z_end')],
    'trajectories': ['xyz_start', 'xyz_end'],
    'vertices': [('x_vert', 'y_vert', 'z_vert')],
    'mc_hdr': [('x_vert', 'y_vert', 'z_vert')],
    'mc_stack': [],
}


def read_tables(source):
    """Load one source file at a time; unsupported tables fail rather than vanish."""
    unknown = set(source) - set(REQUIRED) - OPTIONAL
    if unknown or not set(REQUIRED) <= set(source):
        raise ValueError(f'Unsupported input tables: unknown={unknown}, missing={set(REQUIRED)-set(source)}')
    tables = {}
    for name, dataset in source.items():
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1 or not dataset.dtype.names:
            raise ValueError(f'{name} must be a one-dimensional compound dataset')
        fields = REQUIRED.get(name, {'event_id', 'vertex_id', 'traj_id', 'file_traj_id'}
                              if name == 'mc_stack' else {'event_id', 'vertex_id', 'x_vert', 'y_vert', 'z_vert'})
        if not fields <= set(dataset.dtype.names):
            raise ValueError(f'Missing fields in {name}: {fields-set(dataset.dtype.names)}')
        for field in dataset.dtype.names:
            if field in {'event_id', 'vertex_id', 'segment_id', 'traj_id', 'file_traj_id', 'parent_id'}:
                if dataset.dtype[field].kind not in 'iu':
                    raise ValueError(f'{name}.{field} must be integer')
        if name in OPTIONAL and len(dataset) == 0:
            continue  # larnd-sim expects generator tables to be populated or absent.
        tables[name] = dataset[:]
    return tables


def events(tables):
    """Index once per table, avoiding a full-file Boolean scan for every event."""
    indexes = {}
    for name, rows in tables.items():
        order = np.argsort(rows['event_id'], kind='stable')
        ids, starts, counts = np.unique(rows['event_id'][order], return_index=True, return_counts=True)
        indexes[name] = {int(e): order[start:start+count]
                         for e, start, count in zip(ids, starts, counts)}
    for event_id in sorted(indexes['segments']):
        yield event_id, {name: rows[indexes[name].get(event_id, np.array([], dtype=int))].copy()
                         for name, rows in tables.items()}


def put_integer(rows, field, values):
    """Check before casting, including narrow and unsigned output ID fields."""
    values = [int(v) for v in values]
    limits = np.iinfo(rows.dtype[field])
    if any(v < limits.min or v > limits.max for v in values):
        raise ValueError(f'ID overflow in {field}')
    rows[field] = values


def sentinel(value, dtype):
    return int(value) < 0 or (dtype.kind == 'u' and int(value) == np.iinfo(dtype).max)


def remap_event(event, event_id, trajectory_offset, segment_offset, vertex_offset, shift):
    """Preserve all rows; remap event-local parents and file-level truth links.

    Local trajectory IDs may repeat across vertices, but file_traj_id must be
    unambiguous within this event. Negative/unsigned-max unmatched MC IDs survive.
    """
    result = {name: rows.copy() for name, rows in event.items()}
    trajectories, vertices = event['trajectories'], event['vertices']
    if not len(trajectories) or not len(vertices):
        raise ValueError('Selected event lacks trajectories or vertices')
    vertex_map = {int(old): vertex_offset+i for i, old in enumerate(np.unique(vertices['vertex_id']))}
    local_map, file_map = {}, {}
    for i, row in enumerate(trajectories):
        key = (int(row['vertex_id']), int(row['traj_id']))
        old_file = int(row['file_traj_id'])
        if key in local_map or old_file in file_map or key[1] < 0 or old_file < 0:
            raise ValueError('Ambiguous or invalid trajectory IDs')
        local_map[key] = i
        file_map[old_file] = (i, trajectory_offset+i, key)
    segment_ids = event['segments']['segment_id']
    if len(np.unique(segment_ids)) != len(segment_ids):
        raise ValueError('Duplicate segment IDs in source event')
    for name, rows in result.items():
        put_integer(rows, 'event_id', [event_id] * len(rows))
        try:
            put_integer(rows, 'vertex_id', [vertex_map[int(v)] for v in event[name]['vertex_id']])
        except KeyError as exc:
            raise ValueError(f'Dangling vertex reference in {name}') from exc
        if ('traj_id' in rows.dtype.names) != ('file_traj_id' in rows.dtype.names):
            raise ValueError(f'{name}: traj_id and file_traj_id must occur together')
        if 'file_traj_id' in rows.dtype.names:
            local_ids, file_ids = [], []
            for old in event[name]:
                old_file = int(old['file_traj_id'])
                old_local = int(old['traj_id'])
                missing = sentinel(old_file, rows.dtype['file_traj_id'])
                if name == 'mc_stack' and missing and sentinel(old_local, rows.dtype['traj_id']):
                    local_ids.append(old_local)
                    file_ids.append(old_file)
                    continue
                if missing or old_file not in file_map:
                    raise ValueError(f'Dangling file_traj_id in {name}: {old_file}')
                local, global_id, key = file_map[old_file]
                if key != (int(old['vertex_id']), old_local):
                    raise ValueError(f'Inconsistent trajectory references in {name}')
                local_ids.append(local)
                file_ids.append(global_id)
            put_integer(rows, 'traj_id', local_ids)
            put_integer(rows, 'file_traj_id', file_ids)
        if name == 'trajectories':
            parents = []
            for old in trajectories:
                parent = int(old['parent_id'])
                if sentinel(parent, rows.dtype['parent_id']):
                    parents.append(parent)
                else:
                    key = (int(old['vertex_id']), parent)
                    if key not in local_map:
                        raise ValueError(f'Dangling parent_id: {key}')
                    parents.append(local_map[key])
            put_integer(rows, 'parent_id', parents)
        if name == 'segments':
            put_integer(rows, 'segment_id', range(segment_offset, segment_offset+len(rows)))
        for field in POSITIONS[name]:
            if isinstance(field, tuple):
                for axis, column in enumerate(field):
                    rows[column] += shift[axis]
            else:
                rows[field] += np.asarray(shift)
    maps = dict(vertices=[[old, new] for old, new in vertex_map.items()],
                trajectories=[[old, key[1], key[0], new, local]
                              for old, (local, new, key) in file_map.items()],
                segments=[[int(old), segment_offset+i] for i, old in enumerate(segment_ids)])
    return result, maps, len(vertex_map)


class OutputFileWriter:
    """A failure leaves an incomplete .partial; completed filenames never overwrite."""
    def __init__(self, path, schema, metadata):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + '.partial')
        if self.path.exists():
            raise FileExistsError(self.path)
        self.file = h5py.File(self.partial, 'x')
        self.count = self.trajectory_count = self.segment_count = self.vertex_count = 0
        self.schema = schema
        self.file.attrs.update(status='incomplete', muon_skim_schema_version=1)
        for name, dtype in schema.items():
            self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
        provenance = self.file.create_group('provenance')
        provenance.attrs['run_json'] = json.dumps(metadata, sort_keys=True)
        provenance.attrs['id_map_columns_json'] = json.dumps(dict(
            vertices=['source_vertex_id', 'output_vertex_id'],
            trajectories=['source_file_traj_id', 'source_traj_id', 'source_vertex_id',
                          'output_file_traj_id', 'output_traj_id'],
            segments=['source_segment_id', 'output_segment_id']))
        provenance.create_group('sources')
        provenance.create_dataset('events', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

    def append(self, event, source, source_index, source_info, original_id, choice):
        rows, maps, nvertices = remap_event(event, self.count, self.trajectory_count,
                                           self.segment_count, self.vertex_count, choice['shift_cm'])
        sources = self.file['provenance/sources']
        key = str(source_index)
        if key not in sources:
            group = sources.create_group(key)
            group.attrs['fingerprint_json'] = json.dumps(source_info, sort_keys=True)
            group.create_group('root_attributes').attrs.update(source.attrs)
            for name in event:
                group.create_group(name).attrs.update(source[name].attrs)
        for name, data in rows.items():
            dataset = self.file[name]
            start = len(dataset)
            dataset.resize((start + len(data),))
            dataset[start:] = data
        record = dict(source_index=source_index, source_event_id=int(original_id),
                      output_event_id=self.count, **choice, id_maps=maps)
        records = self.file['provenance/events']
        records.resize((self.count+1,))
        records[self.count] = json.dumps(record, sort_keys=True)
        self.count += 1
        self.trajectory_count += len(rows['trajectories'])
        self.segment_count += len(rows['segments'])
        self.vertex_count += nvertices

    def finish(self):
        # A source can have MC tables globally but none for selected events.
        for name in OPTIONAL:
            if name in self.file and not len(self.file[name]):
                del self.file[name]
        self.file.attrs.update(status='complete', selected_events=self.count)
        self.file.close()
        os.link(self.partial, self.path)  # atomic, fails if destination exists
        self.partial.unlink()
        return dict(path=str(self.path), events=self.count)

    def close_incomplete(self):
        if self.file.id.valid:
            self.file.close()
