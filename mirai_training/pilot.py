"""Prepare complete patient-disjoint pilot exams; outcomes require explicit input."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from .dicom_conversion import candidate_reason, resolve_dicom

KEYS = ['empi_anon', 'acc_anon']
VIEWS = [('R', 'CC'), ('R', 'MLO'), ('L', 'CC'), ('L', 'MLO')]
CLINICAL = KEYS + ['study_date_anon', 'desc', 'GENDER_DESC', 'asses', 'path_severity',
                   'procdate_anon', 'pdate_anon', 'age_at_study', 'tissueden']
IMAGES = KEYS + ['study_date_anon', 'FinalImageType', 'ViewPosition', 'ImageLateralityFinal',
                 'spot_mag', 'anon_dicom_path', 'ManufacturerModelName', 'PatientSex']


def ordered(items, seed):
    return sorted(items, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).digest())


def clinical_candidates(c):
    c = c.copy()
    c['_date'] = pd.to_datetime(c.study_date_anon, format='mixed', errors='coerce').dt.normalize()
    c['_proc'] = pd.to_datetime(c.procdate_anon, format='mixed', errors='coerce').dt.normalize()
    c['_report'] = pd.to_datetime(c.pdate_anon, format='mixed', errors='coerce').dt.normalize()
    c['_severity'] = pd.to_numeric(c.path_severity, errors='coerce')
    c['_screen'] = c.desc.str.contains('screen', case=False) & c.GENDER_DESC.str.casefold().eq('female')
    c['_negative'] = c.asses.isin(['N', 'B']) & c.path_severity.eq('')
    c['_known'] = c.asses.eq('K')
    exams = c.groupby(KEYS).agg(date=('_date', 'min'), dates=('_date', 'nunique'),
        screen=('_screen', 'all'), negative=('_negative', 'all'), known=('_known', 'any'))
    cancer = c[c._severity.isin([0, 1])]
    valid = cancer._proc.notna() & cancer._date.notna() & cancer._proc.ge(cancer._date)
    valid &= (cancer._proc - cancer._date).dt.days.le(365)
    valid &= cancer._report.notna() & cancer._report.ge(cancer._proc)
    valid &= (cancer._report-cancer._proc).dt.days.le(365)
    uncertain = set(cancer.loc[~valid, 'empi_anon'])
    first = cancer.loc[valid].groupby('empi_anon')._proc.min().to_dict()
    # A known-cancer assessment without a dateable cancer endpoint is unresolved.
    uncertain |= set(c.loc[c._known, 'empi_anon']) - set(first)
    last_negative = exams.loc[exams.screen & exams.negative & exams.dates.eq(1)].reset_index().groupby('empi_anon').date.max().to_dict()
    identities = c.groupby('acc_anon').empi_anon.agg(set).to_dict()
    result = {}
    for (pid, eid), row in exams.iterrows():
        if not row.screen or row.dates != 1 or row.known or pd.isna(row.date):
            continue
        if pid in uncertain or len(identities[eid]) != 1:
            continue
        event = first.get(pid)
        if event is not None and event <= row.date:
            continue
        # These are review candidates, never training labels.
        future = event is not None and 0 < (event-row.date).days < 5*365.25
        follow = last_negative.get(pid)
        if not future and (follow is None or (follow-row.date).days < 5*365.25):
            continue
        result[(pid, eid)] = {'index_date': row.date, 'candidate_event_date': event,
                             'candidate_last_negative_screen': follow, 'candidate_positive': future}
    return result, identities


def collect_images(path, candidates, identities):
    groups = {}
    rows = 0
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, usecols=IMAGES, chunksize=50000):
        rows += len(chunk)
        # Check identity across both complete tables, including nonselected images.
        for pid, eid in chunk[KEYS].drop_duplicates().itertuples(index=False, name=None):
            identities.setdefault(eid, set()).add(pid)
        chunk = chunk[chunk.FinalImageType.eq('2D') & chunk.PatientSex.eq('F')]
        for values in chunk.itertuples(index=False, name=None):
            row = dict(zip(chunk.columns, values))
            key = (row['empi_anon'], row['acc_anon'])
            if key not in candidates or candidate_reason(row):
                continue
            slot = (row['ImageLateralityFinal'], row['ViewPosition'])
            slots = groups.setdefault(key, {})
            if slot not in slots:
                slots[slot] = row
            elif slots[slot] is not None:
                compared = ['anon_dicom_path', 'study_date_anon', 'ManufacturerModelName', 'PatientSex']
                if any(slots[slot][name] != row[name] for name in compared):
                    slots[slot] = None  # repeated acquisition or conflicting metadata: exclude
        if rows % 250000 == 0:
            print(f'Image metadata: {rows:,} rows inspected', flush=True)
    return groups, rows


def prepare_pilot(args):
    if args.patients < 20 or not 3 <= args.positive_candidates <= args.patients-3:
        raise ValueError('Require >=20 patients and at least 3 candidates in each outcome stratum')
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Pilot output directory must be new or empty, to preserve existing selections')
    c = pd.read_csv(args.clinical_csv, dtype=str, keep_default_na=False, usecols=CLINICAL)
    candidates, identities = clinical_candidates(c)
    print(f'Clinical candidate exams for review: {len(candidates):,}', flush=True)
    groups, image_rows = collect_images(args.image_metadata_csv, candidates, identities)
    exclusions = Counter()
    pools = {True: [], False: []}
    seen = set()
    # Patient hash avoids simply taking the first cohort. Prefer earliest eligible exam per patient.
    keys = sorted(groups, key=lambda k: (hashlib.sha256(f'{args.seed}:{k[0]}'.encode()).digest(),
                                        candidates[k]['index_date'], k[1]))
    for key in keys:
        pid, eid = key
        if pid in seen:
            continue
        slots = groups[key]
        if len(identities[eid]) != 1:
            exclusions['identity_conflict'] += 1
            continue
        if set(slots) != set(VIEWS) or any(r is None for r in slots.values()):
            exclusions['incomplete_or_ambiguous_views'] += 1
            continue
        dates = pd.to_datetime([r['study_date_anon'] for r in slots.values()], format='mixed', errors='coerce').normalize()
        if dates.isna().any() or any(d != candidates[key]['index_date'] for d in dates):
            exclusions['acquisition_and_clinical_date_mismatch'] += 1
            continue
        try:
            paths = [resolve_dicom(slots[v]['anon_dicom_path'], args.dicom_root, args.strip_prefix) for v in VIEWS]
            if len(set(paths)) != 4:
                raise ValueError('Same source used for multiple views')
        except (ValueError, OSError):
            exclusions['missing_or_conflicting_dicom'] += 1
            continue
        seen.add(pid)
        pools[candidates[key]['candidate_positive']].append(key)
        if len(pools[True]) >= args.positive_candidates and len(pools[False]) >= args.patients-args.positive_candidates:
            break
    requested = {True: args.positive_candidates, False: args.patients-args.positive_candidates}
    for event, count in requested.items():
        if len(pools[event]) < count:
            raise ValueError(f'Insufficient complete, existing-image candidates: positive={len(pools[True])}, other={len(pools[False])}; exclusions={dict(exclusions)}')
    selected = []
    split = {}
    for event, count in requested.items():
        sample = ordered(pools[event][:count], args.seed+1)
        ndev = max(1, round(count*.1))
        ntest = max(1, round(count*.1))
        for i, key in enumerate(sample):
            split[key] = 'dev' if i < ndev else ('test' if i < ndev+ntest else 'train')
        selected.extend(sample)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'selected_images.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=IMAGES); writer.writeheader()
        for key in selected:
            for view in VIEWS:
                writer.writerow(groups[key][view])
    patients = {k[0] for k in selected}
    c[c.empi_anon.isin(patients)].to_csv(out/'clinical_subset.csv', index=False)
    review = []
    for key in selected:
        info = candidates[key]
        date = lambda d: '' if d is None or pd.isna(d) else d.strftime('%Y-%m-%d')
        review.append({'patient_id': key[0], 'exam_id': key[1], 'index_date': date(info['index_date']),
                      'event_date': '', 'last_followup_date': '', 'split_group': split[key],
                      'candidate_event_date': date(info['candidate_event_date']),
                      'candidate_last_negative_screen': date(info['candidate_last_negative_screen']),
                      'label_status': 'REQUIRES_OUTCOME_REVIEW'})
    pd.DataFrame(review).to_csv(out/'outcomes_to_review.csv', index=False)
    policy = getattr(args, 'label_policy', 'review')
    if policy == 'recorded-screening':
        provisional = pd.DataFrame(review)
        provisional['event_date'] = provisional['candidate_event_date']
        provisional['last_followup_date'] = provisional['candidate_last_negative_screen']
        provisional['label_status'] = 'PILOT_RECORDED_EVENTS_AND_NEGATIVE_SCREENING_NOT_ADJUDICATED'
        provisional.to_csv(out/'outcomes_pilot.csv', index=False)
    config = {'image_metadata_csv': str(out/'selected_images.csv'), 'dicom_column': 'anon_dicom_path',
              'strip_prefix': args.strip_prefix, 'dicom_root': str(Path(args.dicom_root).resolve()),
              'output_dir': str(out/'converted'), 'dcmtk': 'dcmj2pnm', 'limit': 0}
    (out/'conversion.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    devices = Counter(r['ManufacturerModelName'] for key in selected if split[key]=='train' for r in groups[key].values())
    from .schema_constants import DEVICE_TO_ID
    known_devices = {DEVICE_TO_ID[name] for name in devices if name in DEVICE_TO_ID}
    report = {'patients': len(selected), 'exams': len(selected), 'images': 4*len(selected),
              'splits': dict(Counter(split.values())), 'candidate_positive_exams': args.positive_candidates,
              'train_image_device_models': dict(devices), 'seed': args.seed, 'image_rows_scanned': image_rows,
              'stage2_has_two_known_device_classes': len(known_devices) >= 2,
              'label_policy': policy,
              'exclusions': dict(exclusions), 'training_ready': False,
              'note': 'Enriched engineering pilot, not population performance evaluation. Recorded-event/negative-screen labels are provisional, miss outside-system events, and require formal follow-up adjudication. No PNG pixels checked yet.'}
    (out/'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def finalize_pilot(args):
    from analysis.mirai_embed.adapt_embed import export
    root = Path(args.pilot_dir).resolve()
    approved = pd.read_csv(args.outcomes_csv, dtype=str, keep_default_na=False)
    planned = pd.read_csv(root/'outcomes_to_review.csv', dtype=str, keep_default_na=False)
    keys = ['patient_id', 'exam_id', 'index_date', 'split_group']
    if set(map(tuple, approved[keys].values)) != set(map(tuple, planned[keys].values)):
        raise ValueError('Reviewed outcomes must retain all selected patients/exams/index dates/splits')
    result = export(args.outcomes_csv, root/'converted/images.csv', root/'metadata.csv', check_images=True)
    base = Path(__file__).resolve().parent.parent/'configs/mirai.yaml'
    c = yaml.safe_load(base.read_text(encoding='utf-8'))
    c['output_dir'] = str(root/'training')
    c['data'].update(metadata_csv=str(root/'metadata.csv'), clinical_csv=str(root/'clinical_subset.csv'),
                     image_metadata_csv=None, risk_factors_json=None)
    c['stage1'].update(epochs=1, batch_size=1, accumulate=4)
    c['stage2'].update(epochs=1, batch_size=16, accumulate=1)
    c['pilot'] = {'outcomes_csv': str(Path(args.outcomes_csv).resolve()),
                  'selection': json.loads((root/'selection.json').read_text()),
                  'for_pipeline_validation_only': True}
    (root/'train.yaml').write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')
    print(json.dumps(result, indent=2))
