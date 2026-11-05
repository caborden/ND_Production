import gc
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import weakref

import h5py
import numpy as np
import nbformat

from event_display import load_event, save_bundle, load_bundle, summary, figure
from reproduce_larnd import instrument, signals_lines


class DiagnosticsTests(unittest.TestCase):
    def test_observer_preserves_arguments_lifetimes_and_error(self):
        records, calls, references = [], [], []
        def zeros(shape, *args, **kwargs):
            calls.append((shape, args, kwargs))
            if len(calls) == 3:
                raise MemoryError('synthetic OOM')
            result = np.zeros(shape, *args, **kwargs)
            references.append(weakref.ref(result))
            return result
        pool = SimpleNamespace(used_bytes=lambda: 100, total_bytes=lambda: 200,
                               free_bytes=lambda: 100, get_limit=lambda: 0)
        cp = SimpleNamespace(zeros=zeros, dtype=np.dtype, get_default_memory_pool=lambda: pool,
                             cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (300, 500))))
        source = '''
def run():
    ievd=i_mod=start_pix=0
    stop_pix=2
    max_pixels=np.array([2])
    max_signal_time=1.0
    for unused in range(3):
        signals = cp.zeros((2, 3, 4), dtype=np.float32)
run()
'''
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'sim.py'
            script.write_text(source)
            code, observer = instrument(cp, script, records.append)
            namespace = dict(cp=cp, np=np, detector=SimpleNamespace(TIME_SAMPLING=.1),
                             __diagnostic_signals_zeros=observer)
            with self.assertRaisesRegex(MemoryError, 'synthetic OOM'):
                exec(code, namespace)
        self.assertEqual([r['phase'] for r in records],
                         ['before', 'allocated', 'before', 'allocated', 'before', 'error'])
        self.assertEqual(records[0]['previous_signals_bytes'], 0)
        self.assertEqual(records[2]['previous_signals_bytes'], 96)
        self.assertEqual(records[-1]['requested_bytes'], 96)
        self.assertEqual(calls, [((2, 3, 4), (), {'dtype': np.float32})] * 3)
        self.assertIs(cp.zeros, zeros)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))

    def test_reject_changed_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'sim.py'
            script.write_text('signals = something_else()')
            with self.assertRaises(ValueError):
                signals_lines(script)

    def test_extract_display_and_bundle(self):
        dtype = [('event_id', 'i8'), ('pdg_id', 'i4')]
        dtype += [(f'{a}_{end}', 'f8') for end in ('start', 'end') for a in 'xyz']
        segments = np.zeros(3, dtype=dtype)
        segments['event_id'] = [17, 17, 18]
        segments['pdg_id'] = [13, 11, 13]
        for a in 'xyz':
            segments[f'{a}_start'] = [.5, -1, .5]
            segments[f'{a}_end'] = [.7, 2, .7]
        record = dict(source_index=0, output_event_id=17, source_event_id=1001740,
                      shift_cm=[1, 2, 3], id_maps={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skim = root / 'skim.hdf5'
            with h5py.File(skim, 'w') as f:
                f['segments'] = segments
                p = f.create_group('provenance')
                p.attrs['run_json'] = json.dumps(dict(active_bounds_cm=[[0, 1]] * 3))
                p.create_dataset('events', data=[json.dumps(record)], dtype=h5py.string_dtype())
                p.create_group('sources/0').attrs['fingerprint_json'] = json.dumps(dict(path='/original'))
            (root / 'files.json').write_text(json.dumps([dict(index=62, skim=str(skim))]))
            event = load_event(root, 62, 17)
            self.assertEqual(len(event['segments']), 2)
            self.assertEqual(summary(event)['endpoint_active'], 1)
            plot = figure(event)
            self.assertEqual(len(plot.data), 3)
            self.assertEqual(len(figure(event, active_only=True).data), 2)
            self.assertEqual(list(plot.layout.scene.xaxis.range), [-2, 3])
            plot.to_json()  # Validate Plotly serialization without a browser.
            bundle = root / 'bundle.hdf5'
            save_bundle(event, bundle)
            loaded = load_bundle(bundle)
            np.testing.assert_array_equal(loaded['segments'], event['segments'])
            self.assertEqual(loaded['metadata'], event['metadata'])
            with self.assertRaises(FileExistsError):
                save_bundle(event, bundle)

    def test_notebook_schema_and_code(self):
        notebook = nbformat.read(Path(__file__).with_name('inspect_events.ipynb'), as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<notebook cell>', 'exec')


if __name__ == '__main__':
    unittest.main()
