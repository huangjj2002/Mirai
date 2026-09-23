"""EMBED audit and explicit-outcome adapter for the upstream Mirai CSV loader.

Audit outputs are candidates, NOT a final training cohort. Export requires a
separate exam outcome table; absence of pathology never creates negative labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import pandas as pd

KEYS = ['patient_id', 'exam_id']
VIEWS = {('L', 'CC'), ('L', 'MLO'), ('R', 'CC'), ('R', 'MLO')}
CSV_COLUMNS = KEYS + ['laterality', 'view', 'file_path', 'years_to_cancer',
                      'years_to_last_followup', 'split_group']
YEAR_DAYS = 365.25


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def date_column(series):
    return pd.to_datetime(series, errors='coerce', format='mixed')


def audit(clinical_path, metadata_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c, m = read(clinical_path), read(metadata_path)
    clinical_rows, metadata_rows = len(c), len(m)
    c = c.rename(columns={'empi_anon': 'patient_id', 'acc_anon': 'exam_id'})
    m = m.rename(columns={'empi_anon': 'patient_id', 'acc_anon': 'exam_id',
                          'ImageLateralityFinal': 'laterality', 'ViewPosition': 'view'})
    for table in (c, m):
        table['date'] = date_column(table.study_date_anon)
    c['severity'] = pd.to_numeric(c.path_severity, errors='coerce')
    c['procedure_date'] = date_column(c.procdate_anon)
    c['report_date'] = date_column(c.pdate_anon)
    c['screening'] = c.desc.str.contains('screen', case=False, na=False)
    c['negative_assessment'] = c.asses.isin(['N', 'B'])
    c['known_malignancy'] = c.asses.eq('K')
    c['special_pathology'] = c.severity.eq(5)
    exams = c.groupby(KEYS, sort=False).agg(
        index_date=('date', 'min'), clinical_date_count=('date', 'nunique'),
        all_screening=('screening', 'all'), all_negative_assessment=('negative_assessment', 'all'),
        known_malignancy=('known_malignancy', 'any'), special_pathology=('special_pathology', 'any'),
        finding_rows=('exam_id', 'size'))
    md = m.groupby(KEYS, sort=False).agg(image_date=('date', 'min'), image_date_count=('date', 'nunique'))
    exams = exams.join(md, how='left')
    exams['date_consistent'] = (exams.clinical_date_count.eq(1) & exams.image_date_count.eq(1)
                                 & exams.index_date.eq(exams.image_date))

    # One patient/exam identity must agree across both source tables.
    identity = pd.concat([c[KEYS], m[KEYS]]).drop_duplicates().groupby('exam_id').patient_id.nunique()
    bad_ids = set(identity[identity.gt(1)].index)
    exams['identity_conflict'] = exams.index.get_level_values('exam_id').isin(bad_ids)

    image_ok = (m.FinalImageType.eq('2D') & m.laterality.isin(['L', 'R'])
                & m.view.isin(['CC', 'MLO']) & m.spot_mag.isin(['', '0', '0.0'])
                & m.PatientSex.eq('F') & m.anon_dicom_path.ne(''))
    images = m.loc[image_ok].drop_duplicates(KEYS + ['anon_dicom_path']).copy()
    counts = images.groupby(KEYS + ['laterality', 'view']).size().unstack(['laterality', 'view'], fill_value=0)
    counts = counts.reindex(columns=pd.MultiIndex.from_tuples(sorted(VIEWS)), fill_value=0)
    exams['has_four_views'] = counts.gt(0).all(axis=1).reindex(exams.index, fill_value=False)
    exams['exactly_four_views'] = counts.eq(1).all(axis=1).reindex(exams.index, fill_value=False)

    events = c.loc[c.severity.isin([0, 1])].copy()
    events['date_quality'] = 'candidate'
    events.loc[events.procedure_date.isna(), 'date_quality'] = 'missing_procedure_date'
    events.loc[events.procedure_date.lt(events.date), 'date_quality'] = 'procedure_before_source_exam'
    events.loc[events.date.isna(), 'date_quality'] = 'missing_source_exam_date'
    signature = ['patient_id', 'procdate_anon', 'bside', 'type'] + [f'path{i}' for i in range(1, 11)]
    events = events.drop_duplicates(signature)
    first_event = events.loc[events.date_quality.eq('candidate')].groupby('patient_id').procedure_date.min()
    exams['first_recorded_cancer_procedure'] = exams.index.get_level_values('patient_id').map(first_event)
    last_record = c.groupby('patient_id').date.max()
    exams['last_record_date_NOT_followup'] = exams.index.get_level_values('patient_id').map(last_record)
    negative = exams[exams.all_screening & exams.all_negative_assessment & exams.date_consistent
                     & ~exams.known_malignancy & ~exams.special_pathology]
    last_negative = negative.reset_index().groupby('patient_id').index_date.max()
    exams['last_negative_screen_candidate'] = exams.index.get_level_values('patient_id').map(last_negative)
    exams['prior_or_same_day_recorded_cancer'] = exams.first_recorded_cancer_procedure.le(exams.index_date)
    candidates = exams.loc[exams.all_screening & exams.date_consistent & ~exams.identity_conflict
                           & exams.has_four_views & ~exams.prior_or_same_day_recorded_cancer].copy()
    candidates['label_status'] = 'UNADJUDICATED_NO_TRAINING_LABEL'
    # Keep duplicate-view candidates visible. Never choose a random first/last image.
    image_candidates = images.merge(candidates.reset_index()[KEYS], on=KEYS, validate='many_to_one')
    candidates.to_csv(output / 'candidate_exams.csv', date_format='%Y-%m-%d')
    image_candidates[KEYS + ['laterality', 'view', 'study_date_anon', 'anon_dicom_path',
                            'ProtocolName', 'SeriesNumber', 'SeriesTime']].to_csv(output / 'candidate_images.csv', index=False)
    events[KEYS + ['procdate_anon', 'pdate_anon', 'bside', 'type', 'path_severity', 'date_quality']
           + [f'path{i}' for i in range(1, 11)]].to_csv(output / 'candidate_cancer_events.csv', index=False)
    # The source outcome table is deliberately empty; raw records do not prove full follow-up.
    pd.DataFrame(columns=KEYS + ['index_date', 'event_date', 'last_followup_date', 'split_group']).to_csv(
        output / 'outcomes_template.csv', index=False)
    summary = {
        'clinical_rows': clinical_rows, 'metadata_rows': metadata_rows,
        'clinical_patients': int(c.patient_id.nunique()), 'clinical_exams': len(exams),
        'identity_conflict_exam_ids': len(bad_ids),
        'screening_four_view_candidate_exams': len(candidates),
        'candidate_patients': int(candidates.reset_index().patient_id.nunique()),
        'exactly_one_per_view_candidate_exams': int(candidates.exactly_four_views.sum()),
        'candidate_images': len(image_candidates),
        'candidate_cancer_event_signatures': len(events),
        'candidate_cancer_patients': int(events.patient_id.nunique()),
        'events_requiring_date_review': int(events.date_quality.ne('candidate').sum()),
        'source_sha256': {str(clinical_path): sha256(clinical_path), str(metadata_path): sha256(metadata_path)},
        'training_ready': False,
        'limitations': ['No image pixels inspected; spot_mag blank is only a selection convention.',
                       'Prior cancer history, DCIS/LCIS endpoint, abnormal-exam resolution and follow-up require study rules.',
                       'Last record and last negative screening are candidates, not verified continuous cancer-free follow-up.',
                       'Duplicate standard views retained for explicit image selection; no random selection.',
                       'No y or negative labels generated from missing pathology.']}
    (output / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def encode_outcome(index_date, event_date, last_followup_date):
    """Legacy Mirai bins: floor(elapsed days / 365.25), event index 0..4.

    last_followup_date is a study-defined observation endpoint, not automatically
    max(study_date). Same-day/past events excluded from this future-risk adapter.
    """
    index = pd.Timestamp(index_date)
    if pd.isna(index):
        raise ValueError('Missing index_date')
    event = pd.Timestamp(event_date) if event_date else None
    follow = pd.Timestamp(last_followup_date) if last_followup_date else None
    if event is not None and (pd.isna(event) or event <= index):
        raise ValueError('Event must be strictly after index exam')
    if follow is not None and (pd.isna(follow) or follow < index):
        raise ValueError('Follow-up cannot precede index exam')
    event_bin = math.floor((event - index).days / YEAR_DAYS) if event is not None else 100
    observed_years = math.floor((follow - index).days / YEAR_DAYS) if follow is not None else 0
    # An observed later cancer gives an observation endpoint under standard survival assumptions.
    if event is not None:
        observed_years = max(observed_years, math.floor((event - index).days / YEAR_DAYS))
    if event_bin >= 5 and observed_years < 1:
        raise ValueError('No observed 1-5 year outcome: censored before one year')
    return event_bin, observed_years


def export(outcomes_path, images_path, destination, check_images=False):
    outcomes, images = read(outcomes_path), read(images_path)
    required = KEYS + ['index_date', 'event_date', 'last_followup_date', 'split_group']
    if not set(required).issubset(outcomes.columns):
        raise ValueError(f'Outcome columns required: {required}')
    if not set(KEYS + ['laterality', 'view', 'file_path']).issubset(images.columns):
        raise ValueError('Image manifest must contain patient_id, exam_id, laterality, view, file_path')
    if outcomes.empty or outcomes[KEYS].eq('').any().any() or outcomes.duplicated(KEYS).any():
        raise ValueError('Outcomes must be nonempty with unique, nonempty patient/exam keys')
    if not outcomes.split_group.isin(['train', 'dev', 'test']).all():
        raise ValueError('Mirai splits must be train/dev/test')
    if outcomes.groupby('patient_id').split_group.nunique().gt(1).any():
        raise ValueError('Patient overlaps multiple splits')
    bins = [encode_outcome(r.index_date, r.event_date, r.last_followup_date) for r in outcomes.itertuples()]
    outcomes['years_to_cancer'] = [x[0] for x in bins]
    outcomes['years_to_last_followup'] = [x[1] for x in bins]
    for r in outcomes.itertuples():
        group = images[(images.patient_id == r.patient_id) & (images.exam_id == r.exam_id)]
        if len(group) != 4 or set(zip(group.laterality, group.view)) != VIEWS:
            raise ValueError(f'{r.patient_id}/{r.exam_id}: require exactly one L/R CC/MLO PNG')
        for path in group.file_path:
            if not (Path(path).is_absolute() or PureWindowsPath(path).is_absolute()) or not path.lower().endswith('.png'):
                raise ValueError(f'Expected absolute PNG path: {path}')
            if not path.isascii():
                raise ValueError('Upstream CSV loader strips non-ASCII path characters; use ASCII PNG paths')
            if check_images and not Path(path).is_file():
                raise ValueError(f'Image is missing: {path}')
    extra = [k for k in ['source_dicom_path', 'device_model'] if k in images]
    result = outcomes.merge(images[KEYS + ['laterality', 'view', 'file_path'] + extra], on=KEYS, validate='one_to_many')
    if result.file_path.duplicated().any():
        raise ValueError('A PNG appears in multiple image slots/exams')
    result = result[CSV_COLUMNS + extra].sort_values(KEYS + ['laterality', 'view'])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return {'exams': len(outcomes), 'image_rows': len(result), 'pixels_checked': False,
            'image_file_existence_checked': check_images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    a = commands.add_parser('audit')
    a.add_argument('--clinical', required=True)
    a.add_argument('--metadata', required=True)
    a.add_argument('--output', required=True)
    e = commands.add_parser('export')
    e.add_argument('--outcomes', required=True)
    e.add_argument('--images', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--check-images', action='store_true')
    args = parser.parse_args()
    if args.command == 'audit':
        summary = audit(args.clinical, args.metadata, args.output)
    else:
        summary = export(args.outcomes, args.images, args.output, args.check_images)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
