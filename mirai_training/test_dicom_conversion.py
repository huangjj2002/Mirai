import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .dicom_conversion import (FLAGS, PRESENTATION_SOP, candidate_reason, check_header,
                               convert_table, render_one, resolve_dicom, verify_png)


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'images/image.dcm'
        self.source.parent.mkdir()
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = PRESENTATION_SOP
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(self.source), {}, file_meta=meta, preamble=b'\0'*128)
        ds.SOPClassUID = PRESENTATION_SOP
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PresentationIntentType = 'FOR PRESENTATION'
        ds.Rows, ds.Columns, ds.SamplesPerPixel = 3, 4, 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.ImageLaterality, ds.ViewPosition = 'R', 'CC'
        ds.ManufacturerModelName = 'Selenia Dimensions'
        ds.PixelData = np.arange(12, dtype=np.uint16).tobytes()
        ds.save_as(self.source, enforce_file_format=True)
        self.ds = ds
        self.row = dict(empi_anon='001', acc_anon='900000000000000001', FinalImageType='2D',
                        ViewPosition='CC', ImageLateralityFinal='R', spot_mag='',
                        anon_dicom_path='images/image.dcm', ManufacturerModelName='Selenia Dimensions',
                        has_pix_array='False', PNG_flipped='True')

    def test_path_mapping_and_selection(self):
        self.assertEqual(resolve_dicom('images/image.dcm', self.root), self.source)
        self.assertEqual(resolve_dicom('/old/images/image.dcm', self.root, '/old'), self.source)
        for raw, prefix in [('../escape', None), ('/other/image', '/old')]:
            with self.assertRaises(ValueError):
                resolve_dicom(raw, self.root, prefix)
        self.assertIsNone(candidate_reason(self.row))  # legacy has_pix_array is ignored
        self.assertEqual(candidate_reason(dict(self.row, FinalImageType='3D')), 'not_2D')
        self.assertEqual(check_header(self.ds, self.row), (4, 3))
        self.ds.PresentationIntentType = 'FOR PROCESSING'
        with self.assertRaises(ValueError):
            check_header(self.ds, self.row)

    def test_png16_validation_resume_and_failed_renderer(self):
        destination = self.root/'out.png'
        def render(command, **kwargs):
            self.assertEqual(command[1:3], FLAGS)
            Image.fromarray(np.arange(12, dtype=np.uint16).reshape(3,4)*100).save(command[-1])
            return SimpleNamespace(returncode=0, stderr='')
        with patch('mirai_training.dicom_conversion.subprocess.run', side_effect=render):
            self.assertEqual(render_one(self.source, destination, (4,3), 'fake', 'test'), 'converted')
            self.assertEqual(render_one(self.source, destination, (4,3), 'fake', 'test'), 'reused')
            with self.assertRaises(ValueError):
                render_one(self.source, destination, (4,3), 'fake', 'changed-version')
        Image.fromarray(np.arange(12, dtype=np.uint8).reshape(3,4)).save(self.root/'8.png')
        with self.assertRaises(ValueError):
            verify_png(self.root/'8.png', (4,3))
        with patch('mirai_training.dicom_conversion.subprocess.run', return_value=SimpleNamespace(returncode=1, stderr='decoder failed')):
            with self.assertRaises(RuntimeError):
                render_one(self.source, self.root/'failed.png', (4,3), 'fake', 'test')
        self.assertFalse((self.root/'failed.png').exists())

    def test_real_dicom_header_audit_and_reported_missing_path(self):
        metadata = self.root/'metadata.csv'
        with metadata.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.row)); w.writeheader(); w.writerow(self.row)
            w.writerow(dict(self.row, anon_dicom_path='images/missing.dcm'))
        args = SimpleNamespace(image_metadata_csv=str(metadata), output_dir=str(self.root/'audit'),
                               dicom_column='anon_dicom_path', dicom_root=str(self.root),
                               strip_prefix=None, dcmtk='absent', limit=0, dry_run=True)
        with self.assertRaises(RuntimeError):
            convert_table(args)
        result = json.loads((self.root/'audit/conversion.json').read_text())
        self.assertEqual(result['counts']['headers_ok'], 1)
        self.assertEqual(result['counts']['errors'], 1)
        self.assertTrue(result['scan_complete'])


if __name__ == '__main__':
    unittest.main()
