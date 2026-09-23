"""Behavior checks for raw table discovery; no ML dependencies."""
import csv
import tempfile
import unittest
from pathlib import Path

from .field_audit import factor_mapping, run_audit, scan_table


class FieldAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, rows):
        path = self.root / name
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(rows)
        return path

    def test_blanks_unknowns_quoted_values_and_preview(self):
        p = self.write('clinical.csv', [
            ['empi_anon', 'RACE_DESC', 'tissueden', 'notes'],
            ['001', 'Unknown, Unavailable or Unreported', '0', 'line1\nline2'],
            ['002', 'Caucasian or White', '', 'a,b'],
        ])
        table = scan_table(p)
        fields = {f['name']: f for f in table['fields']}
        self.assertEqual(table['rows_scanned'], 2)
        self.assertTrue(table['complete'])
        self.assertEqual(fields['RACE_DESC']['explicit_unknown_tokens'], 1)
        self.assertEqual(fields['tissueden']['nonblank'], 1)
        self.assertNotIn('categories', fields['empi_anon'])
        self.assertNotIn('categories', fields['notes'])
        self.assertFalse(scan_table(p, 1)['complete'])
        self.assertTrue(scan_table(p, 2)['complete'])

    def test_malformed_rows_fail_instead_of_silently_shifting_columns(self):
        p = self.write('bad.csv', [['a', 'b'], ['1', '2', '3']])
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            scan_table(p)

    def test_full_schema_report_and_missing_source(self):
        clinical = self.write('clinical.csv', [['age_at_study', 'RACE_DESC', 'family_history'], ['51', '', '']])
        metadata = self.write('metadata.csv', [['ManufacturerModelName', 'PatientAge', 'PNG_flipped'], ['New Device', '051Y', 'False']])
        result = run_audit(clinical, metadata, self.root / 'report')
        self.assertEqual(len(result['mirai_factors']), 34)
        self.assertIn('metadata.PatientAge', next(f for f in result['mirai_factors'] if f['factor'] == 'age')['candidate_columns'])
        self.assertIn('clinical.RACE_DESC', next(f for f in result['mirai_factors'] if f['factor'] == 'race')['candidate_columns'])
        self.assertIn('clinical.family_history', result['additional_clinical_candidates'])
        self.assertTrue((self.root / 'report/report.md').is_file())
        self.assertTrue((self.root / 'report/field_audit.json').is_file())
        with self.assertRaises(FileNotFoundError):
            run_audit(clinical, self.root / 'missing.csv', self.root / 'unused')


if __name__ == '__main__':
    unittest.main()
