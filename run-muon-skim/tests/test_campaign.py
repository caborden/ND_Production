"""No scheduler or GPU required: dependency, validation, and retry contracts."""
from contextlib import redirect_stdout
from argparse import Namespace
import io
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import campaign as c
import test_muon_skim as fixtures


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(root=str(self.root), outputs=str(self.root / 'scratch-output'),
                           name='test', gpu_workers=10, flow_workers=5,
                           cpu_account='cpu-account', gpu_account='gpu-account',
                           hashes={}, repos={}, external=[], sources=[], pilot_parts=[])

    def tearDown(self):
        self.temp.cleanup()

    def test_groups_cover_every_file_exactly_once_and_balance(self):
        for count in (0, 3, 200, 203):
            grouped = [c.groups(list(range(count)), 10, i) for i in range(10)]
            self.assertEqual(sorted(v for group in grouped for v in group), list(range(count)))
            self.assertLessEqual(max(map(len, grouped)) - min(map(len, grouped)), 1)

    def test_submit_holds_root_and_wires_all_four_stages(self):
        calls = []
        def sbatch(command, **kwargs):
            calls.append(command)
            return str(100 + len(calls)) + ';cluster\n'
        with patch.object(c.subprocess, 'check_output', side_effect=sbatch), \
                patch.object(c.subprocess, 'run') as run, redirect_stdout(io.StringIO()):
            c.submit(self.config)
        self.assertEqual(len(calls), 4)
        self.assertTrue((Path(self.config['outputs']) / 'locks/.submit.lock').is_file())
        self.assertFalse((self.root / '.submit.lock').exists())
        self.assertIn('--hold', calls[0])
        self.assertIn('--array=0-9%10', calls[1])
        self.assertIn('--array=0-4%5', calls[2])
        for i, command in enumerate(calls):
            self.assertIn('--kill-on-invalid-dep=yes', command)
            self.assertFalse(any(arg.startswith('--mem') for arg in command))
            if i:
                self.assertIn(f'--dependency=afterok:{100+i}', command)
        run.assert_called_once_with(['scontrol', 'release', '101'], check=True)
        self.assertEqual(c.read(self.root / 'submission.json')['attempts'][0]['status'], 'submitted')
        with self.assertRaisesRegex(ValueError, 'Already submitted'):
            c.submit(self.config)

    def test_partial_submission_cancels_already_submitted_jobs(self):
        with patch.object(c.subprocess, 'check_output', side_effect=[
                '101\n', subprocess.CalledProcessError(1, 'sbatch')]), \
                patch.object(c.subprocess, 'run') as run:
            with self.assertRaises(subprocess.CalledProcessError):
                c.submit(self.config)
        run.assert_called_once_with(['scancel', '101'], check=False)
        self.assertEqual(c.read(self.root / 'submission.json')['attempts'][0]['status'], 'submission_failed')

    def test_retry_refuses_live_jobs(self):
        c.write(self.root / 'submission.json', dict(attempts=[dict(jobs=[dict(id='101')])]))
        with patch.object(c.subprocess, 'check_output', return_value='101\n'):
            with self.assertRaisesRegex(ValueError, 'still queued/running'):
                c.submit(self.config, retry=True)

    def test_retry_after_terminal_jobs_submits_new_dependencies(self):
        c.write(self.root / 'submission.json', dict(attempts=[dict(jobs=[dict(id='101')])]))
        with patch.object(c.subprocess, 'check_output', side_effect=['', '201', '202', '203', '204']), \
                patch.object(c.subprocess, 'run') as run, redirect_stdout(io.StringIO()):
            c.submit(self.config, retry=True)
        run.assert_called_once_with(['scontrol', 'release', '201'], check=True)
        self.assertEqual(len(c.read(self.root / 'submission.json')['attempts']), 2)

    def fake_larnd(self, path):
        with h5py.File(path, 'w') as data:
            for name in ('packets', 'mc_packets_assn', 'segments', 'trajectories', 'vertices'):
                data.create_dataset(name, data=np.arange(2))

    def test_success_receipt_reuse_and_input_change_rejection(self):
        source = self.root / 'source'
        source.write_text('input')
        output = self.root / 'out.h5'
        def execute(command, *args):
            self.fake_larnd(command[0])
        def run():
            c.run_file(source, output, lambda p: [p], self.root / 'log', self.root, c.larnd_check)
        with patch.object(c, 'execute', side_effect=execute) as runner:
            run()
            run()
            self.assertEqual(runner.call_count, 1)
            source.write_text('changed input')
            with self.assertRaisesRegex(ValueError, 'mismatched completion receipt'):
                run()

    def test_failed_output_not_published_and_partial_preserved_on_retry(self):
        source = self.root / 'source'
        source.write_text('input')
        output = self.root / 'out.h5'
        def broken(command, *args):
            Path(command[0]).write_text('broken hdf5')
        with patch.object(c, 'execute', side_effect=broken):
            with self.assertRaises(OSError):
                c.run_file(source, output, lambda p: [p], 'log', self.root, c.larnd_check)
        self.assertFalse(output.exists())
        with patch.object(c, 'execute', side_effect=lambda command, *args: self.fake_larnd(command[0])):
            c.run_file(source, output, lambda p: [p], 'log', self.root, c.larnd_check)
        self.assertEqual(len(list(self.root.glob('*.partial.failed-*'))), 1)
        self.assertTrue(output.exists())

    def test_unreceipted_final_is_preserved_and_regenerated(self):
        source = self.root / 'source'
        source.write_text('input')
        output = self.root / 'out.h5'
        self.fake_larnd(output)
        with patch.object(c, 'execute', side_effect=lambda command, *args: self.fake_larnd(command[0])) as run:
            c.run_file(source, output, lambda p: [p], 'log', self.root, c.larnd_check)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(len(list(self.root.glob('out.h5.failed-*'))), 1)

    def test_flow_does_not_start_with_missing_gpu_output(self):
        c.write(self.root / 'files.json', [dict(skim='missing', larnd='missing', flow='missing')])
        with patch.object(c, 'run_file') as run:
            with self.assertRaises(FileNotFoundError):
                c.flow_stage(self.config, 0)
            run.assert_not_called()

    def test_changed_code_guard(self):
        code = self.root / 'code.py'
        code.write_text('old')
        self.config['hashes'][str(code)] = c.sha(code)
        c.guard(self.config)
        code.write_text('new')
        with self.assertRaisesRegex(ValueError, 'Frozen campaign'):
            c.guard(self.config)

    def test_stage_script_is_valid_bash_and_uses_correct_environment(self):
        config = dict(self.config, work='/path with spaces', envs=dict(
            skim='/skim/bin/python', analysis='/analysis environment'))
        script = self.root / 'stage.sh'
        script.write_text(c.stage_script(config))
        subprocess.run(['bash', '-n', str(script)], check=True)
        self.assertIn('H5FLOW_NOMPI=1', script.read_text())
        self.assertIn('SLURM_CPUS_PER_TASK', script.read_text())

    def test_worker_lock_on_scratch_excludes_other_process_and_releases(self):
        lock = Path(self.config['outputs']) / 'locks/.skim-0.lock'
        contender = [sys.executable, '-c', '''
import fcntl, sys
with open(sys.argv[1], 'a') as stream:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(23)
''', str(lock)]
        def stage(config):
            self.assertTrue(lock.is_file())
            self.assertFalse((self.root / '.skim-0.lock').exists())
            self.assertEqual(subprocess.run(contender).returncode, 23)
        with patch.object(c, 'skim_stage', side_effect=stage):
            c.run_stage(self.config, 'skim', 0)
        self.assertEqual(subprocess.run(contender).returncode, 0)

    def test_worker_lock_released_after_stage_failure(self):
        with patch.object(c, 'skim_stage', side_effect=RuntimeError('stage failed')):
            with self.assertRaisesRegex(RuntimeError, 'stage failed'):
                c.run_stage(self.config, 'skim', 0)
        with patch.object(c, 'skim_stage') as stage:
            c.run_stage(self.config, 'skim', 0)
            stage.assert_called_once()

    def test_real_skim_gate_reuse_and_duplicate_content(self):
        fixture = fixtures.SkimTests()
        fixture.setUp()
        try:
            stage = Path(c.__file__).parent
            (self.root / 'code').mkdir()
            shutil.copytree(stage / 'muon_skim', self.root / 'code/muon_skim')
            c.write(self.root / 'skim-config.json', fixture.config)
            self.config.update(outputs=str(self.root / 'output'),
                               sources=[c.identity(fixture.source)], pilot_source_hashes=[],
                               seed_offset=40, larnd=str(fixture.repo), envs=dict(skim=sys.executable))
            with redirect_stdout(io.StringIO()):
                c.skim_stage(self.config)
            rows = c.read(self.root / 'files.json')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['index'], 40)
            with patch.object(c, 'execute') as run:
                c.skim_stage(self.config)
                run.assert_not_called()
            self.config['pilot_source_hashes'] = [c.sha(fixture.source)]
            with self.assertRaisesRegex(ValueError, 'Duplicate original source content'):
                c.skim_stage(self.config)
        finally:
            fixture.tearDown()

    def test_prepare_excludes_pilot_freezes_settings_and_never_submits(self):
        work = self.root / 'work'
        jobs = work / 'muon-skim-jobs'
        scratch = self.root / 'scratch'
        production = work / 'ND_Production-muon-skim/run-muon-skim'
        shutil.copytree(Path(c.__file__).parent / 'muon_skim', production / 'muon_skim')
        track_code = jobs / 'track-selection-29cm-v1/code'
        track_code.mkdir(parents=True)
        for name in c.TRACK_FILES:
            (track_code / name).write_text('# frozen selection fixture\n')
        source_dir = self.root / 'sources'
        source_dir.mkdir()
        for i in range(6):
            (source_dir / f'{i}.EDEPSIM.hdf5').write_text(str(i))
        source = source_dir / '0.EDEPSIM.hdf5'
        (source_dir / 'alias.EDEPSIM.hdf5').symlink_to(source)
        larnd = work / 'larnd-sim'
        (larnd / 'larndsim/config').mkdir(parents=True)
        response = self.root / 'response'
        response.write_text('response fixture')
        registry = larnd / 'larndsim/config/config.yaml'
        registry.write_text(yaml.safe_dump(dict(fsd_cube=dict(LIGHT_SIMULATED=False, RESPONSE=str(response)))))
        flow = work / 'flow-fsdcube-916fb2e1/ndlar_flow'
        pedestal = self.root / 'pedestal.json'
        pedestal.write_text('{}')
        calib = flow / 'yamls/fsdcube_flow/reco/charge/CalibHitBuilderMC.yaml'
        calib.parent.mkdir(parents=True)
        calib.write_text(yaml.safe_dump(dict(params=dict(pedestal_file=str(pedestal)))))
        for name in c.WORKFLOWS:
            path = flow / 'yamls/fsdcube_flow/workflows' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in ['skim.venv/bin/python', 'larnd.venv/bin/simulate_pixels.py',
                     'flow-fsdcube-916fb2e1/flow.venv/bin/h5flow', 'analysis/bin/python']:
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.chmod(0o755)
        (jobs / 'larnd-sim.commit.txt').write_text(c.FLOW_COMMIT)
        pilot_part = self.root / 'pilot.tracks.hdf5'
        with h5py.File(pilot_part, 'w') as data:
            data.attrs.update(status='complete', schema_version='2.0')
            data.require_group('sources/0').attrs['status'] = 'complete'
            meta = data.require_group('sources/0/metadata').attrs
            meta.update(min_length_cm=29, is2x2=False, input_file='/pilot/FLOW.hdf5')
            for name in ['track_selection_fsd.py', 'efield.py', 'track_output_hdf5.py']:
                meta[name + '_sha256'] = c.sha(track_code / name)
            data.create_dataset('tracks/data', data=np.array([(30.,), (31.,)], dtype=[('length_cm', 'f8')]))
        tracks_manifest = scratch / 'fsd-muon-production/track-selection/FSDCubeMuons_pilot_v1.min29cm/parts/batch_manifest.json'
        c.write(tracks_manifest, dict(status='complete', min_length_cm=29, files=[dict(
            status='complete', output=str(pilot_part), input='/pilot/FLOW.hdf5')]))
        output_base = scratch / 'output'
        manifest = output_base / 'run-muon-skim/FSDCubeMuons_pilot_v1/EDEPSIM_H5/0000000/FSDCubeMuons_pilot_v1.0000000/manifest.json'
        config = c.read(Path(c.__file__).parent / 'configs/fsd_cube.json')
        c.write(manifest, dict(status='complete', sources=[dict(path=str(source), sha256=c.sha(source))],
                              outputs=[dict(path='pilot')], metadata=dict(config=config,
                              geometry_sources=[dict(path=str(registry), sha256=c.sha(registry))],
                              code=[dict(path=str(p), sha256=c.sha(p)) for p in (production / 'muon_skim').glob('*.py')])))
        args = Namespace(work_root=str(work), name='overnight', additional_sources=5,
                         gpu_workers=10, flow_workers=5, cpu_account='cpu', gpu_account='gpu',
                         source_dir=str(source_dir), analysis_env=str(work / 'analysis'))
        with patch.dict(c.os.environ, SCRATCH=str(scratch), ND_PRODUCTION_OUTDIR_BASE=str(output_base)), \
                patch.object(c, 'git_state', return_value=dict(commit=c.FLOW_COMMIT, diff_sha256='clean')), \
                patch.object(c.subprocess, 'run') as run, redirect_stdout(io.StringIO()):
            c.prepare(args)
            run.assert_not_called()
            prepared = c.read(jobs / 'overnight/campaign.json')
            c.guard(prepared)
        self.assertEqual(len(prepared['sources']), 5)
        self.assertNotIn(str(source), [p['path'] for p in prepared['sources']])
        self.assertEqual(prepared['pilot_tracks'], 2)
        self.assertTrue((Path(prepared['outputs']) / 'locks').is_dir())
        self.assertEqual(c.read(jobs / 'overnight/skim-config.json'), config)
        self.assertEqual((jobs / 'overnight/campaign.sha256').read_text().strip(),
                         c.sha(jobs / 'overnight/campaign.json'))


if __name__ == '__main__':
    unittest.main()
