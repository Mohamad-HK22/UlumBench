import json
import tempfile
import unittest
from pathlib import Path
from ulumbench.protocol import load_config, make_plan
from ulumbench.data import Dataset


class ProtocolTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps({'experiment': 'state_assistance', 'repetitions': 0, 'max_tokens': 12}))
            with self.assertRaises(ValueError):
                load_config(p)

    def test_traversal_is_rejected(self):
        data = Dataset.__new__(Dataset)
        with tempfile.TemporaryDirectory() as tmp:
            data.root = Path(tmp)
            with self.assertRaises(ValueError):
                data.path('../private')

    def test_empty_diagnostic_plan_and_bad_limit(self):
        class Data:
            families = {}; fingerprint = 'synthetic'
        config = {'experiment': 'state_assistance', 'repetitions': 1}
        requests, report = make_plan(Data(), config)
        self.assertEqual(requests, [])
        self.assertEqual(report['selected_candidate_calls'], 0)
        with self.assertRaises(ValueError):
            make_plan(Data(), config, 0)
