import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from optics_graph.legacy import EXPECTED_NAMES, legacy_manifest, precision_progress, read_legacy_evidence, run_legacy


class LegacyAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        layout = self.root / 'Private_Project_Materials/Research/Closed_Model_Full_Experiment_20261027/answer_first.json'
        layout.parent.mkdir(parents=True)
        layout.write_text(json.dumps({'components': [{'id': n, 'x_mm': 1, 'y_mm': 2, 'z_mm': 100} for n in sorted(EXPECTED_NAMES)]}))
        archive = self.root / 'Private_Project_Materials/Research/Vision_Michelson_Assembly'
        archive.mkdir(parents=True)
        (archive / 'result.json').write_text(json.dumps({'source_sha256': {'scene.py': 'historic'},
            'source_layout_sha256': 'historic-layout', 'status': 'completed', 'success': True,
            'verification': {'accepted': True}, 'seed': 0}))
        (archive / 'calibration.json').write_text('{}')
        source = self.root / 'robot-arm-control/simulation/scene.py'
        source.parent.mkdir(parents=True)
        source.write_text('source')

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_tracks_input_and_transforms(self):
        first = legacy_manifest(self.root)
        self.assertEqual(len(first['targets']), 15)
        self.assertEqual(first['targets'][0]['world_position_m'], [-.149, -.098, .7999999999999999])
        path = Path(first['source_path'])
        path.write_text(path.read_text() + ' ')
        self.assertNotEqual(first['source_sha256'], legacy_manifest(self.root)['source_sha256'])

    def test_existing_output_is_rejected_before_execution(self):
        output = self.root / 'existing'
        output.mkdir()
        with patch('subprocess.Popen') as launch:
            with self.assertRaises(FileExistsError):
                run_legacy(self.root, output)
            launch.assert_not_called()

    def test_historical_evidence_is_explicit(self):
        result = read_legacy_evidence(self.root)
        self.assertTrue(result['historical'])
        self.assertEqual(result['status'], 'historical_accepted')
        self.assertEqual(result['source_sha256'], 'historic-layout')
        self.assertIn('not a new run', result['scope'])

    def test_mismatched_layout_hash_rejects_reported_success(self):
        output = self.root / 'fresh'
        manifest = legacy_manifest(self.root)
        def finish(timeout):
            (output / 'trajectory.npz').write_bytes(b'trajectory')
            (output / 'world_model.mjb').write_bytes(b'model')
            import hashlib
            digest = lambda v: hashlib.sha256(v).hexdigest()
            report = {'status': 'completed', 'success': True, 'verification': {'accepted': True},
                      'source_sha256': manifest['code_sha256'],
                      'calibration_sha256': manifest['calibration_sha256'],
                      'source_layout_sha256': 'wrong-layout',
                      'public_scene': {'target_frame': {'source_sha256': manifest['source_sha256']}},
                      'trajectory_sha256': digest(b'trajectory'), 'world_model_sha256': digest(b'model')}
            (output / 'result.json').write_text(json.dumps(report))
            return 0
        with patch('subprocess.Popen') as launch:
            launch.return_value.wait.side_effect = finish
            result = run_legacy(self.root, output)
        self.assertFalse(result['success'])
        self.assertFalse(result['bindings_verified'])

    def test_missing_result_is_not_success(self):
        with patch('subprocess.Popen') as launch:
            launch.return_value.wait.return_value = 0
            result = run_legacy(self.root, self.root / 'fresh')
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('fresh result', result['failure'])

    def test_live_motion_labels_separate_rehearsal_from_execution(self):
        output = self.root / 'simulation'
        output.mkdir()
        (output / 'result.json').write_text(json.dumps({
            'status': 'running', 'recovery_actions': ['replan_supported_turn_lift']}))
        log = output.parent / 'resume-worker.log'
        log.write_text('  10.0s  SH0: yaw at support\n')
        self.assertEqual(precision_progress(output)['motion_phase'], {
            'object_id':'SH0', 'phase':'yaw at support', 'simulation_s':10., 'rehearsal':True})
        log.write_text(log.read_text() + 'SH0: supported turn MuJoCo rehearsal passed\n  18.5s  SH0: lift\n')
        self.assertEqual(precision_progress(output)['motion_phase'], {
            'object_id':'SH0', 'phase':'lift', 'simulation_s':18.5, 'rehearsal':False})

if __name__ == '__main__':
    unittest.main()
