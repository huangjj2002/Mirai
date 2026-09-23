import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from .pilot import CLINICAL, IMAGES, VIEWS, clinical_candidates, prepare_pilot, finalize_pilot


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clinical = []
        images = []
        for i in range(25):
            pid, eid = f'p{i:03d}', f'e{i:03d}'
            baseline = dict.fromkeys(CLINICAL, '')
            baseline.update(empi_anon=pid, acc_anon=eid, study_date_anon='2020-01-01',
                desc='Screening mammogram', GENDER_DESC='Female', asses='N', age_at_study='50', tissueden='2.0')
            later = dict(baseline, acc_anon=eid+'later', study_date_anon='2026-01-01')
            if i < 7:
                later.update(desc='Diagnostic mammogram', study_date_anon='2023-01-01', asses='S', path_severity='0.0',
                             procdate_anon='2023-01-10', pdate_anon='2023-01-12')
            self.clinical.extend([baseline, later])
            for side, view in VIEWS:
                relative = f'cohort_1/{pid}/{side}_{view}.dcm'
                path = self.root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'path fixture only; header check happens at conversion')
                images.append(dict(empi_anon=pid, acc_anon=eid, study_date_anon='2020-01-01',
                    FinalImageType='2D', ViewPosition=view, ImageLateralityFinal=side, spot_mag='',
                    anon_dicom_path='/old/'+relative, ManufacturerModelName='Lorad Selenia' if i%2 else 'Selenia Dimensions',
                    PatientSex='F'))
        pd.DataFrame(self.clinical).to_csv(self.root/'clinical.csv', index=False)
        pd.DataFrame(images).to_csv(self.root/'images.csv', index=False)
        self.args = SimpleNamespace(clinical_csv=str(self.root/'clinical.csv'),
            image_metadata_csv=str(self.root/'images.csv'), dicom_root=str(self.root), strip_prefix='/old',
            patients=20, positive_candidates=5, seed=2026, output_dir=str(self.root/'pilot'))

    def test_candidates_are_temporal_and_invalid_event_dates_exclude_patient(self):
        c = pd.DataFrame(self.clinical)
        candidates, _ = clinical_candidates(c)
        self.assertTrue(candidates[('p000', 'e000')]['candidate_positive'])
        self.assertFalse(candidates[('p024', 'e024')]['candidate_positive'])
        c.loc[(c.empi_anon=='p000') & c.acc_anon.str.endswith('later'), 'procdate_anon'] = '1900-01-01'
        candidates, _ = clinical_candidates(c)
        self.assertFalse(any(k[0]=='p000' for k in candidates))

    def test_prepare_and_finalize_require_explicit_outcomes(self):
        prepare_pilot(self.args)
        root = Path(self.args.output_dir)
        selected = pd.read_csv(root/'selected_images.csv', dtype=str)
        self.assertEqual(len(selected), 80)
        self.assertEqual(selected.empi_anon.nunique(), 20)
        review = pd.read_csv(root/'outcomes_to_review.csv', dtype=str, keep_default_na=False)
        self.assertTrue(review.event_date.eq('').all())
        self.assertTrue(review.last_followup_date.eq('').all())
        self.assertEqual(review.groupby('patient_id').split_group.nunique().max(), 1)
        self.assertFalse(json.loads((root/'selection.json').read_text())['training_ready'])
        (root/'converted').mkdir()
        image_manifest = []
        for i, row in selected.iterrows():
            p = root/f'image{i}.png'; p.write_bytes(b'path fixture, not actual PNG')
            image_manifest.append(dict(patient_id=row.empi_anon, exam_id=row.acc_anon,
                laterality=row.ImageLateralityFinal, view=row.ViewPosition, file_path=str(p),
                source_dicom_path=row.anon_dicom_path, device_model=row.ManufacturerModelName))
        pd.DataFrame(image_manifest).to_csv(root/'converted/images.csv', index=False)
        args = SimpleNamespace(pilot_dir=str(root), outcomes_csv=str(root/'outcomes_to_review.csv'))
        with self.assertRaises(ValueError):
            finalize_pilot(args)
        # Only fixture truth supplies labels here. Production never copies candidate dates automatically.
        review['event_date'] = review['candidate_event_date']
        review['last_followup_date'] = review.apply(lambda r: '2026-01-01' if not r.event_date else '', axis=1)
        approved = root/'approved.csv'; review.to_csv(approved, index=False)
        args.outcomes_csv = str(approved)
        finalize_pilot(args)
        output = pd.read_csv(root/'metadata.csv', dtype=str)
        self.assertEqual(len(output), 80)
        self.assertTrue(output.device_model.ne('').all())
        config = yaml.safe_load((root/'train.yaml').read_text())
        self.assertEqual(config['stage1']['epochs'], 1)
        self.assertEqual(config['image']['size'], [1664, 2048])
        with self.assertRaises(ValueError):
            prepare_pilot(self.args)  # never overwrite a selected cohort

    def test_provisional_labels_require_explicit_policy(self):
        self.args.label_policy = 'recorded-screening'
        prepare_pilot(self.args)
        outcomes = pd.read_csv(Path(self.args.output_dir)/'outcomes_pilot.csv', keep_default_na=False)
        self.assertEqual(outcomes.event_date.ne('').sum(), 5)
        self.assertTrue(outcomes.label_status.str.startswith('PILOT_').all())

    def test_only_four_standard_views_and_no_arbitrary_duplicate_choice(self):
        path = self.root/'images.csv'
        images = pd.read_csv(path, dtype=str, keep_default_na=False)
        # A missing RMLO and two different RCC acquisitions both exclude their exams.
        images = images[~(images.empi_anon.eq('p024') & images.ImageLateralityFinal.eq('R') & images.ViewPosition.eq('MLO'))]
        duplicate = images[(images.empi_anon=='p023') & (images.ImageLateralityFinal=='R') & (images.ViewPosition=='CC')].iloc[0].to_dict()
        duplicate['anon_dicom_path'] += '.second'
        conflict = images[images.empi_anon.eq('p022')].iloc[0].to_dict()
        conflict['study_date_anon'] = '2020-02-01'
        extra = []
        template = images.iloc[0].to_dict()
        for view in ['ML', 'XCCL', 'CCID', 'LRR']:
            extra.append(dict(template, ViewPosition=view))
        for kind in ['3D', 'cview', 'ROI_SS', 'ROI_SSC']:
            extra.append(dict(template, FinalImageType=kind))
        pd.concat([images, pd.DataFrame([duplicate, conflict]+extra)], ignore_index=True).to_csv(path, index=False)
        prepare_pilot(self.args)
        selected = pd.read_csv(Path(self.args.output_dir)/'selected_images.csv', dtype=str)
        self.assertFalse(selected.empi_anon.isin(['p022', 'p023', 'p024']).any())
        self.assertEqual(set(zip(selected.ImageLateralityFinal, selected.ViewPosition)), set(VIEWS))
        self.assertTrue(selected.FinalImageType.eq('2D').all())
        self.assertTrue(selected.groupby('empi_anon').size().eq(4).all())


if __name__ == '__main__':
    unittest.main()
