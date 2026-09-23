import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import data_signature, digest
from .models import RiskTargets, risk_schema
from .schema_constants import DEVICE_TO_ID, VIEWS

REQUIRED = ['patient_id', 'exam_id', 'laterality', 'view', 'file_path',
            'years_to_cancer', 'years_to_last_followup', 'split_group']


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def integer(value, name):
    v = float(value)
    if not np.isfinite(v) or not v.is_integer():
        raise ValueError(f'{name} must be an integer, got {value!r}')
    return int(v)


def labels(years_to_cancer, followup):
    if years_to_cancer < 0 or followup < 0:
        raise ValueError('Negative event/followup interval')
    event = years_to_cancer < 5
    time = years_to_cancer if event else min(followup, 5) - 1
    if time < 0:
        return None
    seq = np.zeros(5, dtype=np.float32)
    if event:
        seq[time:] = 1
    return seq, (np.arange(5) <= time).astype(np.float32), int(event), time


def load_exams(c, check_pixels=False):
    """Audit immutable labels supplied by the researcher; never infer cancer-free followup."""
    schema = risk_schema(c)
    groups, patient_splits = defaultdict(list), {}
    path_owners = {}
    for row in rows(c['data']['metadata_csv']):
        if any(k not in row or row[k] is None for k in REQUIRED):
            raise ValueError(f'Metadata requires columns: {REQUIRED}')
        pid, eid, split = row['patient_id'], row['exam_id'], row['split_group']
        if not pid or not eid or '\t' in pid+eid:
            raise ValueError('Patient/exam identifiers must be nonempty and contain no tab')
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f'Invalid split: {split}')
        if pid in patient_splits and patient_splits[pid] != split:
            raise ValueError(f'Patient appears across splits: {pid}')
        patient_splits[pid] = split
        p = Path(row['file_path'])
        if not p.is_absolute() or p.suffix.lower() != '.png':
            raise ValueError('file_path must be an absolute PNG path, not DICOM')
        slot = (pid, eid, row['laterality'], row['view'])
        if str(p) in path_owners and path_owners[str(p)] != slot:
            raise ValueError(f'Image reused under conflicting exam/view: {p}')
        path_owners[str(p)] = slot
        groups[(pid, eid)].append(row)

    devices = {}
    if c['data'].get('image_metadata_csv'):
        for r in rows(c['data']['image_metadata_csv']):
            path, model = r['anon_dicom_path'], r['ManufacturerModelName'].strip()
            if path in devices and devices[path] != model:
                raise ValueError(f'Conflicting device metadata: {path}')
            devices[path] = model
    clinical = defaultdict(lambda: {'age': set(), 'density': set()})
    if c['data'].get('clinical_csv'):
        for r in rows(c['data']['clinical_csv']):
            key = (r['empi_anon'], r['acc_anon'])
            if key not in groups:
                continue
            for key_name, field in [('age', 'age_at_study'), ('density', 'tissueden')]:
                raw = r.get(field, '').strip()
                if raw:
                    try:
                        value = float(raw)
                        if np.isfinite(value):
                            clinical[key][key_name].add(value)
                    except ValueError:
                        pass
    factors = {}
    if c['data'].get('risk_factors_json'):
        with open(c['data']['risk_factors_json'], encoding='utf-8') as f:
            for r in json.load(f):
                key = (str(r['patient_id']), str(r['exam_id']))
                if key in factors:
                    raise ValueError('Duplicate exam in risk_factors_json')
                if set(r['factors']) - {s['name'] for s in schema}:
                    raise ValueError('Unknown risk factor name')
                factors[key] = r['factors']

    exams, excluded, pixel_states = [], [], []
    for key, images in groups.items():
        contract = {(r['years_to_cancer'], r['years_to_last_followup'], r['split_group']) for r in images}
        if len(contract) != 1:
            raise ValueError(f'Conflicting labels in exam {key}')
        slots = [(r['laterality'], r['view']) for r in images]
        if len(slots) != len(set(slots)):
            raise ValueError(f'Duplicate image per view: {key}; select one explicitly')
        if set(slots) != set(VIEWS):
            excluded.append({'patient_id': key[0], 'exam_id': key[1], 'reason': 'incomplete_standard_views'})
            continue
        images = [images[slots.index(v)] for v in VIEWS]
        row = images[0]
        lab = labels(integer(row['years_to_cancer'], 'years_to_cancer'),
                     integer(row['years_to_last_followup'], 'years_to_last_followup'))
        if lab is None:
            excluded.append({'patient_id': key[0], 'exam_id': key[1], 'reason': 'no_supervised_horizon'})
            continue
        invalid = None
        for r in images:
            p = Path(r['file_path'])
            if not p.is_file():
                invalid = f'missing_image:{p}'
                break
            stat = p.stat()
            pixel_states.append((str(p), stat.st_size, stat.st_mtime_ns))
            if check_pixels:
                try:
                    with Image.open(p) as im:
                        if im.mode not in ('I', 'I;16', 'I;16B', 'I;16L'):
                            raise ValueError(f'Expected 16-bit grayscale PNG, got {im.mode}')
                        im.verify()
                except Exception as e:
                    invalid = f'invalid_image:{p}:{e}'
                    break
        if invalid:
            excluded.append({'patient_id': key[0], 'exam_id': key[1], 'reason': invalid})
            continue
        vals, known = [], []
        explicit = factors.get(key, {})
        for s in schema:
            name, size = s['name'], s['size']
            value = explicit.get(name)
            if name not in explicit and key in clinical and name in ('age', 'density'):
                candidates = clinical[key][name]
                if len(candidates) == 1:
                    raw = next(iter(candidates))
                    if name == 'age' and 0 < raw < 120:
                        value = [float(i == np.searchsorted([40,50,60,70,80], raw, side='left')) for i in range(6)]
                    if name == 'density' and raw in (1,2,3,4):
                        value = [float(i == raw-1) for i in range(4)]
            if value is None:
                vals.append(np.zeros(size, dtype=np.float32)); known.append(False)
                continue
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            if len(value) != size or not np.isfinite(value).all() or not np.isin(value, [0,1]).all():
                raise ValueError(f'Invalid encoded factor {name} at {key}: expected {size} binary entries')
            if size > 1 and value.sum() != 1:
                raise ValueError(f'Categorical factor {name} must be one-hot; use null for missing')
            vals.append(value); known.append(True)
        names = []
        for r in images:
            name = r.get('device_model', '').strip()
            mapped = devices.get(r.get('source_dicom_path', ''), '')
            if name and mapped and name != mapped:
                raise ValueError('Direct device_model disagrees with source_dicom_path metadata')
            names.append(name or mapped)
        exams.append({'key': '\t'.join(key), 'patient_id': key[0], 'exam_id': key[1],
                      'split': row['split_group'], 'paths': [r['file_path'] for r in images],
                      'y_seq': lab[0], 'y_mask': lab[1], 'event': lab[2], 'time': lab[3],
                      'factors': vals, 'factor_known': np.array(known),
                      'device_labels': np.array([DEVICE_TO_ID.get(n, 0) for n in names], dtype=np.int64),
                      'device_known': np.array([n in DEVICE_TO_ID for n in names])})
    if not exams:
        raise ValueError('No usable exams. Check input paths, views, and observed outcomes.')
    counts = Counter(e['split'] for e in exams)
    factor_counts = {s['name']: sum(int(e['factor_known'][i]) for e in exams if e['split']=='train')
                     for i,s in enumerate(schema)}
    device_counts = Counter(int(d) for e in exams if e['split']=='train'
                            for d,k in zip(e['device_labels'], e['device_known']) if k)
    report = {'exams_by_split': dict(counts), 'excluded': excluded,
              'train_known_risk_factors': factor_counts, 'train_device_image_counts': dict(device_counts),
              'schema': schema, 'source_hashes': data_signature(c),
              'image_stat_fingerprint': digest(sorted(pixel_states)),
              'pixels_verified': check_pixels}
    return exams, report


def prepare_image(path, size):
    with Image.open(path) as im:
        if im.mode not in ('I', 'I;16', 'I;16B', 'I;16L'):
            raise ValueError(f'Expected PNG16, got {im.mode}: {path}')
        # Original Scale_2d uses PIL I mode and torchvision's bilinear resize.
        im = im.convert('I').resize(tuple(size), Image.Resampling.BILINEAR)
    a = np.asarray(im)
    width = a.shape[1]
    # Original Linux behavior; wide sums avoid NumPy 1.x Windows int32 overflow.
    if a[:, :width-width*3//4].sum(dtype=np.float64) > a[:, width*3//4:].sum(dtype=np.float64):
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return im


def image_tensor(path, cfg, stats, augment=False):
    im = prepare_image(path, cfg['size'])
    if augment:
        if random.random() < .5:
            im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        im = im.rotate(random.randint(-20,20))
    a = np.asarray(im, dtype=np.float32).copy()
    return ((torch.from_numpy(a).unsqueeze(0).expand(3,-1,-1)-stats['mean'])/stats['std']).contiguous()


def compute_stats(exams, c):
    total = square = 0.0
    count = 0
    paths = sorted({p for e in exams if e['split']=='train' for p in e['paths']})
    if not paths:
        raise ValueError('Training images required for normalization')
    for i,p in enumerate(paths):
        a = np.asarray(prepare_image(p, c['image']['size']), dtype=np.float64)
        total += a.sum(); square += np.square(a).sum(); count += a.size
        if (i+1) % 100 == 0:
            print(f'Normalization: {i+1}/{len(paths)} images', flush=True)
    mean = total/count
    std = float(np.sqrt(max(0, square/count - mean**2)))
    if std <= 0:
        raise ValueError('Zero image variance')
    return {'mean': float(mean), 'std': std, 'train_images': len(paths)}


class ExamDataset(Dataset):
    def __init__(self, exams, split, stage, c, stats, features=None):
        self.exams = [e for e in exams if e['split']==split]
        if not self.exams:
            raise ValueError(f'No usable {split} exams')
        self.stage, self.c, self.stats, self.features = stage,c,stats,features
        self.augment = split == 'train' and features is None
        self.entries = [(e,i) for e in self.exams for i in (range(4) if stage==1 else [None])]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e, view = self.entries[idx]
        if self.stage == 1:
            x = image_tensor(e['paths'][view], self.c['image'], self.stats, self.augment)
        elif self.features is not None:
            x = self.features[e['key']]
        else:
            x = torch.stack([image_tensor(p,self.c['image'],self.stats) for p in e['paths']],dim=1)
        return {'x':x, 'key':e['key'], 'y_seq':torch.from_numpy(e['y_seq']),
                'y_mask':torch.from_numpy(e['y_mask']), 'event':e['event'], 'time':e['time'],
                'factors':[torch.from_numpy(v) for v in e['factors']],
                'factor_known':torch.from_numpy(e['factor_known']),
                'device_labels':torch.from_numpy(e['device_labels']),
                'device_known':torch.from_numpy(e['device_known']),
                'time_seq':torch.zeros(4,dtype=torch.long),
                'view_seq':torch.tensor([0,1,0,1]), 'side_seq':torch.tensor([0,0,1,1])}


def batch_inputs(batch, device):
    x = batch['x'].to(device)
    factors = RiskTargets([v.to(device) for v in batch['factors']],batch['factor_known'].to(device))
    model_batch = {k:batch[k].to(device) for k in ('time_seq','view_seq','side_seq')}
    return x, factors, model_batch
