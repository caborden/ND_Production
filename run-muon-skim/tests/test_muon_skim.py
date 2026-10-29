"""CPU-only synthetic geometry, truth integrity, packing, CLI, and failure tests."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from muon_skim.geometry import clip_segments, cube_geometry, load_config, translations, validate_config
from muon_skim.io import events, read_tables, remap_event
from muon_skim.pipeline import discover, run
from muon_skim.selection import choose_translation, qualifying_translations


def fixture(path, count=1, optional=False, length=30.0):
    common = [('event_id', 'u4'), ('vertex_id', 'u8')]
    ids = [('traj_id', 'i4'), ('file_traj_id', 'u4')]
    segment_dtype = common + ids + [('segment_id', 'u4'), ('pdg_id', 'i4')] + [
        (f'{axis}{suffix}', 'f8') for suffix in ('', '_start', '_end') for axis in 'xyz'
    ] + [('dE', 'f8'), ('t0', 'f8')]
    trajectory_dtype = common + ids + [('parent_id', 'i4'), ('pdg_id', 'i4'),
        ('xyz_start', 'f8', (3,)), ('xyz_end', 'f8', (3,)), ('pxyz_start', 'f8', (3,))]
    vertex_dtype = common + [('x_vert', 'f8'), ('y_vert', 'f8'), ('z_vert', 'f8'), ('t_event', 'f8')]
    arrays = {name: [] for name in ('segments', 'trajectories', 'vertices')}
    for event_id in range(count):
        segments = np.zeros(3, dtype=segment_dtype)
        trajectories = np.zeros(3, dtype=trajectory_dtype)
        vertices = np.zeros(2, dtype=vertex_dtype)
        for rows in (segments, trajectories, vertices):
            rows['event_id'] = 100 + event_id * 7
        for rows in (segments, trajectories):
            rows['vertex_id'] = [7, 9, 7]
            rows['traj_id'] = [10, 10, 30]  # local IDs repeat across vertices
            rows['file_traj_id'] = np.array([100, 200, 300]) + event_id * 1000
            rows['pdg_id'] = [13, -13, 11]
        segments['segment_id'] = np.array([22, 44, 88]) + event_id * 100
        segments['x_start'], segments['x_end'] = 20, 20
        segments['z_start'], segments['z_end'] = 20, 20
        segments['y_end'] = [length, length, 1]
        for axis in 'xyz':
            segments[axis] = (segments[f'{axis}_start'] + segments[f'{axis}_end']) / 2
        segments['dE'], segments['t0'] = [50, 51, 3], [0.1, 0.2, 0.3]
        trajectories['parent_id'] = [-1, -1, 10]
        trajectories['xyz_start'] = np.column_stack([segments[f'{a}_start'] for a in 'xyz'])
        trajectories['xyz_end'] = np.column_stack([segments[f'{a}_end'] for a in 'xyz'])
        trajectories['pxyz_start'] = [0, -1000, 0]
        vertices['vertex_id'] = [7, 9]
        for name, rows in zip(arrays, (segments, trajectories, vertices)):
            arrays[name].append(rows)
    with h5py.File(path, 'w') as f:
        f.attrs['source_note'] = 'fixture'
        for name, parts in arrays.items():
            f.create_dataset(name, data=np.concatenate(parts))
            f[name].attrs['units'] = 'cm'
        if optional:
            dtype = common + [('traj_id', 'i4'), ('file_traj_id', 'i4')]
            stack = np.zeros(2, dtype=dtype)
            stack['event_id'], stack['vertex_id'] = 100, 7
            stack['traj_id'], stack['file_traj_id'] = [10, -999], [100, -999]
            f.create_dataset('mc_stack', data=stack)
            f.create_dataset('mc_hdr', data=np.concatenate(arrays['vertices']))


class SkimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.EDEPSIM.hdf5'
        fixture(self.source)
        self.config = load_config(STAGE / 'configs/fsd_cube.json')
        self.config['translation_grid'] = dict(counts=[1, 2, 1], steps_cm=[50, 30, 48])
        self.repo = self.root / 'larnd'
        base = self.repo / 'larndsim'
        for directory in ('config', 'pixel_layouts', 'detector_properties'):
            (base / directory).mkdir(parents=True)
        registry = {'fsd_cube': {'PIXEL_LAYOUT': 'cube.yaml', 'DET_PROPERTIES': 'cube.yaml'}}
        layout = dict(tile_positions={1: [500, 150, 240]}, tile_orientations={1: [-1, 1, -1]},
                      drift_length=500, pixel_pitch=60,
                      chip_channel_to_position={i*5+j: [i, j] for i in range(8) for j in range(5)})
        (base / 'config/config.yaml').write_text(yaml.safe_dump(registry))
        (base / 'pixel_layouts/cube.yaml').write_text(yaml.safe_dump(layout))
        (base / 'detector_properties/cube.yaml').write_text(yaml.safe_dump(dict(tpc_offsets=[[0, 0, 0]])))
        self.bounds, _ = cube_geometry(self.repo, 'fsd_cube')

    def tearDown(self):
        self.temp.cleanup()

    def produce(self, name='out', inputs=None, config=None):
        with redirect_stdout(io.StringIO()):
            return run(inputs or [self.source], self.root / name, config or self.config, self.repo)

    def test_geometry_and_crossing_with_both_endpoints_outside(self):
        np.testing.assert_allclose(self.bounds, [[0, 50], [0, 30], [0, 48]])
        start = np.array([[20, -10, 20], [0, -10, 20], [60, -10, 20]], dtype=float)
        end = start + [0, 50, 0]
        a, b = clip_segments(start, end, self.bounds)
        np.testing.assert_allclose(a, [[20, 0, 20]])
        np.testing.assert_allclose(b, [[20, 30, 20]])

    def test_strict_length_cut_and_muon_identity(self):
        for length, expected in [(28.9, 0), (29.0, 0), (29.1, 1), (30, 1)]:
            fixture(self.source, length=length)
            with h5py.File(self.source) as f:
                candidates = qualifying_translations(f['segments'][:], np.zeros((1, 3)), self.bounds, 29)
            self.assertEqual(len(candidates), expected)
            if expected:
                self.assertEqual(len(candidates[0]['muons']), 2)
        with h5py.File(self.source, 'r+') as f:
            rows = f['segments'][:]
            rows['pdg_id'] = 11
            f['segments'][:] = rows
        self.assertEqual(self.produce()['selected_events'], 0)

    def test_selection_uses_clipped_length_not_full_trajectory(self):
        fixture(self.source, length=100)
        with h5py.File(self.source) as f:
            candidates = qualifying_translations(f['segments'][:], np.zeros((1, 3)), self.bounds, 31)
        self.assertEqual(candidates, [])

    def test_seeded_uniform_translation_choice_not_muon_count(self):
        candidates = [dict(translation_index=i, muons=list(range(i+1))) for i in range(3)]
        picks = [choose_translation(candidates, seed, 'a'*64, 123)['translation_index'] for seed in range(600)]
        counts = np.bincount(picks, minlength=3)
        self.assertTrue(np.all((counts > 150) & (counts < 250)), counts)
        self.assertEqual(choose_translation(candidates, 7, 'a', 3), choose_translation(candidates, 7, 'a', 3))
        self.assertIsNone(choose_translation([], 7, 'a', 3))

    def test_whole_event_translation_truth_maps_and_sentinels(self):
        fixture(self.source, optional=True)
        with h5py.File(self.source) as f:
            _, event = next(events(read_tables(f)))
        shifted, maps, nvertices = remap_event(event, 8, 40, 50, 60, [1, 2, 3])
        self.assertEqual(nvertices, 2)
        np.testing.assert_array_equal(shifted['trajectories']['traj_id'], [0, 1, 2])
        np.testing.assert_array_equal(shifted['trajectories']['file_traj_id'], [40, 41, 42])
        np.testing.assert_array_equal(shifted['trajectories']['parent_id'], [-1, -1, 0])
        np.testing.assert_array_equal(shifted['segments']['vertex_id'], [60, 61, 60])
        np.testing.assert_array_equal(shifted['segments']['segment_id'], [50, 51, 52])
        np.testing.assert_array_equal(shifted['mc_stack']['file_traj_id'], [40, -999])
        np.testing.assert_array_equal(shifted['mc_stack']['traj_id'], [0, -999])
        for name, rows in shifted.items():
            self.assertTrue(np.all(rows['event_id'] == 8))
        np.testing.assert_allclose(shifted['segments']['y_end'], event['segments']['y_end']+2)
        np.testing.assert_allclose(shifted['vertices']['x_vert'], event['vertices']['x_vert']+1)
        np.testing.assert_allclose(shifted['mc_hdr']['z_vert'], event['mc_hdr']['z_vert']+3)
        np.testing.assert_allclose(shifted['trajectories']['xyz_start'], event['trajectories']['xyz_start']+[1, 2, 3])
        for field in ('dE', 't0'):
            np.testing.assert_array_equal(shifted['segments'][field], event['segments'][field])
        np.testing.assert_array_equal(shifted['trajectories']['pxyz_start'], event['trajectories']['pxyz_start'])
        self.assertEqual(maps['segments'][0], [22, 50])

    def test_dangling_and_inconsistent_truth_rejected(self):
        with h5py.File(self.source) as f:
            _, original = next(events(read_tables(f)))
        cases = [('segments', 'file_traj_id', 999), ('segments', 'traj_id', 999),
                 ('segments', 'vertex_id', 999), ('trajectories', 'parent_id', 999)]
        for name, field, value in cases:
            event = deepcopy(original)
            event[name][field][0] = value
            with self.assertRaises(ValueError):
                remap_event(event, 0, 0, 0, 0, [0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'overflow'):
            remap_event(original, 2**32, 0, 0, 0, [0, 0, 0])

    def test_exact_file_size_and_complete_provenance(self):
        fixture(self.source, count=5)
        self.config['events_per_file'] = 2
        result = self.produce()
        self.assertEqual([r['events'] for r in result['outputs']], [2, 2, 1])
        for output in result['outputs']:
            with h5py.File(output['path']) as f:
                self.assertEqual(f.attrs['status'], 'complete')
                self.assertNotIn('mc_hdr', f)
                np.testing.assert_array_equal(f['segments']['segment_id'], np.arange(len(f['segments'])))
                self.assertEqual(len(np.unique(f['trajectories']['file_traj_id'])), len(f['trajectories']))
                record = json.loads(f['provenance/events'][0])
                self.assertIn('source_event_id', record)
                self.assertIn('id_maps', record)
                self.assertEqual(f['provenance/sources/0/root_attributes'].attrs['source_note'], 'fixture')
                self.assertEqual(f['provenance/sources/0/segments'].attrs['units'], 'cm')
                metadata = json.loads(f['provenance'].attrs['run_json'])
                self.assertEqual(metadata['config']['min_length_cm'], 29)
                self.assertTrue(metadata['code'])
        self.assertFalse(list((self.root/'out').glob('*.partial')))
        self.assertEqual(len((self.root/'out/larnd_inputs.txt').read_text().splitlines()), 3)

    def test_empty_selection_publishes_only_audit(self):
        self.config['min_length_cm'] = 31
        result = self.produce()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['outputs'], [])
        self.assertEqual((self.root/'out/larnd_inputs.txt').read_text(), '')
        self.assertFalse(list((self.root/'out').glob('*.hdf5')))

    def test_source_with_zero_events(self):
        with h5py.File(self.source, 'r+') as f:
            for name in list(f):
                dtype = f[name].dtype
                del f[name]
                f.create_dataset(name, shape=(0,), dtype=dtype)
        result = self.produce()
        self.assertEqual(result['scanned_events'], 0)
        self.assertEqual(result['outputs'], [])

    def test_different_optional_schemas_split_output_files(self):
        second = self.root/'second.hdf5'
        fixture(second, optional=True)
        result = self.produce(inputs=[self.source, second])
        self.assertEqual(len(result['outputs']), 2)
        with h5py.File(result['outputs'][0]['path']) as f:
            self.assertIn('mc_stack', f)
        with h5py.File(result['outputs'][1]['path']) as f:
            self.assertNotIn('mc_stack', f)

    def test_empty_optional_datasets_are_omitted(self):
        fixture(self.source, optional=True)
        with h5py.File(self.source, 'r+') as f:
            for name in ('mc_stack', 'mc_hdr'):
                dtype = f[name].dtype
                del f[name]
                f.create_dataset(name, shape=(0,), dtype=dtype)
        result = self.produce()
        with h5py.File(result['outputs'][0]['path']) as f:
            self.assertNotIn('mc_stack', f)
            self.assertNotIn('mc_hdr', f)

    def test_packing_does_not_change_placement(self):
        fixture(self.source, count=3, length=60)
        # Extend below the Cube to make two positive translations qualify.
        with h5py.File(self.source, 'r+') as f:
            rows = f['segments'][:]
            rows['y_start'] = -30
            f['segments'][:] = rows
        first = self.produce('one')
        config = deepcopy(self.config)
        config['events_per_file'] = 1
        second = self.produce('two', config=config)
        def choices(result):
            records = []
            for output in result['outputs']:
                with h5py.File(output['path']) as f:
                    records.extend(json.loads(r)['translation_index'] for r in f['provenance/events'][:])
            return records
        self.assertEqual(choices(first), choices(second))

    def test_discovery_and_no_overwrite(self):
        listing = self.root/'inputs.txt'
        listing.write_text('# relative paths\nsource.EDEPSIM.hdf5\n')
        self.assertEqual(discover(input_list=listing), [self.source])
        self.assertEqual(discover([self.root]), [self.source])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            discover([self.source], [str(self.source)])
        with self.assertRaisesRegex(ValueError, 'No inputs match'):
            discover(patterns=[str(self.root/'absent*')])
        self.produce()
        with self.assertRaises(FileExistsError):
            self.produce()

    def test_failure_keeps_partial_and_marks_manifest(self):
        with h5py.File(self.source, 'r+') as f:
            rows = f['trajectories'][:]
            rows['parent_id'][0] = 999
            f['trajectories'][:] = rows
        with self.assertRaisesRegex(ValueError, 'parent_id'):
            self.produce()
        directory = self.root/'out'
        self.assertEqual(json.loads((directory/'manifest.json').read_text())['status'], 'failed')
        self.assertFalse(list(directory.glob('*.hdf5')))
        self.assertFalse((directory/'larnd_inputs.txt').exists())
        with h5py.File(next(directory.glob('*.partial'))) as f:
            self.assertEqual(f.attrs['status'], 'incomplete')

    def test_invalid_config_and_unknown_tables_fail(self):
        for key, value in [('min_length_cm', float('nan')), ('seed', -1), ('events_per_file', 0)]:
            config = deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        with h5py.File(self.source, 'r+') as f:
            f.create_dataset('unknown_truth', data=[1])
        with self.assertRaisesRegex(ValueError, 'unknown_truth'):
            self.produce()

    def test_cli_overrides_and_empty_exit_success(self):
        output = self.root/'cli'
        result = subprocess.run([sys.executable, '-m', 'muon_skim', str(self.source),
            '--larnd-repo', str(self.repo), '--output-dir', str(output), '--min-length-cm', '100',
            '--seed', '9', '--events-per-file', '2'], cwd=STAGE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output/'manifest.json').read_text())['metadata']
        self.assertEqual(metadata['config']['seed'], 9)
        self.assertEqual(metadata['config']['events_per_file'], 2)
        self.assertEqual(metadata['config']['min_length_cm'], 100)

    def test_notebook_executes_preview_and_provenance_using_package(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Matplotlib is optional for notebook inspection')
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.produce()
        notebook = json.loads((STAGE/'inspect_muon_skim.ipynb').read_text())
        namespace = {}
        with patch.object(Path, 'cwd', return_value=STAGE), redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] != 'code':
                    continue
                exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)
                if index == 2:
                    namespace.update(larnd_repo=self.repo, source_path=self.source,
                                     skim_directory=self.root/'out', config=self.config)
        self.assertIsNotNone(namespace['choice'])
        self.assertEqual(namespace['qualifying_translations'], qualifying_translations)
        plt.close('all')

    def test_production_wrapper_with_stubbed_cluster_helpers(self):
        # Exercise the real wrapper/CLI; replace only cluster setup and timing.
        stage = self.root/'production/run-muon-skim'
        stage.mkdir(parents=True)
        util = stage.parent/'util'
        util.mkdir()
        (util/'reload_in_container.inc.sh').write_text(':\n')
        (util/'init.inc.sh').write_text('''globalIdx=0000000
subDir=0000000
outDir="$ND_PRODUCTION_OUTDIR_BASE/run-muon-skim/$ND_PRODUCTION_OUT_NAME"
outName="$ND_PRODUCTION_OUT_NAME.$globalIdx"
run() { "$@"; }
''')
        shutil.copy(STAGE/'run_muon_skim.sh', stage)
        config_path = self.root/'wrapper-config.json'
        config_path.write_text(json.dumps(self.config))
        env = dict(os.environ, PYTHONPATH=str(STAGE), ND_PRODUCTION_INDEX='0',
                   ND_PRODUCTION_OUT_NAME='pilot', ND_PRODUCTION_OUTDIR_BASE=str(self.root/'output'),
                   ND_PRODUCTION_MUON_SKIM_INPUT_FILE=str(self.source),
                   ND_PRODUCTION_MUON_SKIM_LARND_REPO=str(self.repo),
                   ND_PRODUCTION_MUON_SKIM_CONFIG=str(config_path),
                   ND_PRODUCTION_MUON_SKIM_PYTHON=sys.executable)
        result = subprocess.run(['bash', 'run_muon_skim.sh'], cwd=stage, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root/'output/run-muon-skim/pilot/EDEPSIM_H5/0000000/pilot.0000000'
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['selected_events'], 1)
        self.assertEqual(manifest['metadata']['config']['seed'], self.config['seed'])


if __name__ == '__main__':
    unittest.main()
