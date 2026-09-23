"""Contract tests execute upstream CSV dataset code, without loading the network."""
import ast
import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from adapt_embed import encode_outcome, export

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'code/Mirai-master/Mirai-master/onconet/datasets'


def upstream_loader():
    # Execute original source method bodies, not a reimplementation of its labels.
    abstract_tree = ast.parse((UPSTREAM / 'abstract_onco_dataset.py').read_text())
    abstract_class = next(n for n in abstract_tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in abstract_class.body if isinstance(n, ast.FunctionDef) and n.name == 'image_paths_by_views')
    abstract_class.body = [method]
    abstract_class.bases = []
    abstract_class.decorator_list = []
    tree = ast.parse((UPSTREAM / 'csv_mammo_cancer.py').read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    klass.decorator_list = []
    pad = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pad_to_length')
    module = ast.Module(body=[abstract_class, klass, pad], type_ignores=[])
    namespace = {'np': np, 'defaultdict': defaultdict, 'tqdm': SimpleNamespace(tqdm=lambda x: x),
                 'MAX_TIME': 10, 'MAX_VIEWS': 2, 'MAX_SIDES': 2}
    exec(compile(ast.fix_missing_locations(module), str(UPSTREAM), 'exec'), namespace)
    return namespace[klass.name]


class ContractTests(unittest.TestCase):
    def test_year_boundaries_and_censoring(self):
        self.assertEqual(encode_outcome('2020-01-01', '2020-12-31', ''), (0, 0))
        self.assertEqual(encode_outcome('2020-01-01', '2021-01-01', ''), (1, 1))
        self.assertEqual(encode_outcome('2020-01-01', '', '2022-10-01'), (100, 2))
        with self.assertRaises(ValueError):
            encode_outcome('2020-01-01', '', '2020-12-01')
        with self.assertRaises(ValueError):
            encode_outcome('2020-01-01', '2020-01-01', '')

    def test_export_runs_original_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outcomes = pd.DataFrame([
                ['p1', '900000000000000001', '2020-01-01', '2022-03-01', '', 'test'],
                ['p2', '900000000000000002', '2020-01-01', '', '2022-10-01', 'test'],
            ], columns=['patient_id', 'exam_id', 'index_date', 'event_date', 'last_followup_date', 'split_group'])
            images = pd.DataFrame([
                [r.patient_id, r.exam_id, side, view, str(tmp / f'{r.patient_id}_{side}_{view}.png')]
                for r in outcomes.itertuples() for side in ['L', 'R'] for view in ['CC', 'MLO']
            ], columns=['patient_id', 'exam_id', 'laterality', 'view', 'file_path'])
            images['source_dicom_path'] = [f'images/source_{i}.dcm' for i in range(len(images))]
            images['device_model'] = 'Selenia Dimensions'
            outcomes.to_csv(tmp / 'outcomes.csv', index=False)
            images.to_csv(tmp / 'images.csv', index=False)
            result = export(tmp / 'outcomes.csv', tmp / 'images.csv', tmp / 'mirai.csv')
            self.assertEqual(result['image_rows'], 8)
            exported = pd.read_csv(tmp / 'mirai.csv', dtype=str)
            self.assertEqual(set(exported.source_dicom_path), set(images.source_dicom_path))
            self.assertTrue(exported.device_model.eq('Selenia Dimensions').all())
            cls = upstream_loader()
            loader = cls.__new__(cls)
            loader.args = SimpleNamespace(max_followup=5, num_images=4, is_ccds_server=False,
                                          use_c_view_if_available=False)
            with (tmp / 'mirai.csv').open() as stream:
                loader.metadata_json = list(csv.DictReader(stream))
            dataset = loader.create_dataset('test', '')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]['y_seq'].tolist(), [0, 0, 1, 1, 1])
            self.assertEqual(dataset[0]['y_mask'].tolist(), [1, 1, 1, 0, 0])
            self.assertEqual(dataset[1]['y_seq'].tolist(), [0, 0, 0, 0, 0])
            self.assertEqual(dataset[1]['y_mask'].tolist(), [1, 1, 0, 0, 0])
            self.assertEqual(dataset[0]['side_seq'].tolist(), [0, 0, 1, 1])
            self.assertEqual(dataset[0]['view_seq'].tolist(), [0, 1, 0, 1])
            self.assertIn('900000000000000001', dataset[0]['exam'])
            # Missing or repeated views must not silently become a complete exam.
            images.iloc[:-1].to_csv(tmp / 'images_bad.csv', index=False)
            with self.assertRaises(ValueError):
                export(tmp / 'outcomes.csv', tmp / 'images_bad.csv', tmp / 'bad.csv')
            # Same patient in two partitions must fail before producing a CSV.
            outcomes.loc[1, 'patient_id'] = 'p1'
            outcomes.loc[1, 'split_group'] = 'train'
            outcomes.to_csv(tmp / 'outcomes_bad.csv', index=False)
            with self.assertRaises(ValueError):
                export(tmp / 'outcomes_bad.csv', tmp / 'images.csv', tmp / 'bad.csv')


if __name__ == '__main__':
    unittest.main(verbosity=2)
