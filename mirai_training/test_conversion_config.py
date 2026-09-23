import json
from pathlib import Path
import tempfile
import unittest

from .cli import conversion_args, parser


class ConversionConfigTests(unittest.TestCase):
    def test_relative_paths_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'conversion.json'
            config.write_text(json.dumps({'image_metadata_csv': 'table.csv', 'output_dir': 'png',
                'dicom_root': 'dicom', 'strip_prefix': '/mnt/NAS2/mammo/anon_dicom', 'limit': 16}))
            args = conversion_args(parser().parse_args(['convert-dicom', '--config', str(config), '--limit', '0']))
            self.assertEqual(args.limit, 0)
            self.assertEqual(args.output_dir, str((root/'png').resolve()))
            self.assertEqual(args.strip_prefix, '/mnt/NAS2/mammo/anon_dicom')
            self.assertEqual(args.dicom_column, 'anon_dicom_path')

    def test_legacy_command_and_missing_inputs(self):
        args = conversion_args(parser().parse_args(['convert-dicom', '--image-metadata-csv', 'table.csv', '--output-dir', 'png']))
        self.assertEqual(args.limit, 16)
        with self.assertRaises(ValueError):
            conversion_args(parser().parse_args(['convert-dicom']))


if __name__ == '__main__':
    unittest.main()
