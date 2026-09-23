import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
PATH_KEYS = ('metadata_csv', 'risk_factors_json', 'image_metadata_csv', 'clinical_csv')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        c = yaml.safe_load(f)
    if not isinstance(c, dict):
        raise ValueError('Configuration must be a mapping')
    for key in PATH_KEYS:
        value = c['data'].get(key)
        c['data'][key] = str((path.parent / value).resolve()) if value else None
    c['output_dir'] = str((path.parent / c['output_dir']).resolve())
    c['_config_path'] = str(path)
    c.setdefault('synthetic', False)
    if c['model']['hidden_dim'] not in (512, 1024) or c['model']['num_heads'] not in (8, 16):
        raise ValueError('Use the original search space: hidden_dim 512/1024, num_heads 8/16')
    if c['model']['pool'] not in ('Simple_AttentionPool', 'GlobalMaxPool', 'GlobalAvgPool'):
        raise ValueError('Unknown original aggregation pool')
    if c['image']['size'] != [1664, 2048] and not c['synthetic']:
        raise ValueError('Real runs retain original [width,height]=[1664,2048] resolution')
    for stage in ('stage1', 'stage2'):
        s = c[stage]
        if s['batch_size'] < 1 or s['accumulate'] < 1 or s['epochs'] < 1:
            raise ValueError(f'Invalid training sizes: {stage}')
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu for local checks.')
    major, minor = torch.cuda.get_device_capability()
    arch = f'sm_{major}{minor}'
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f'This PyTorch build does not list {arch}; install a compatible CUDA build. '
                           'Use --device cpu for local checks.')
    return torch.device(name if name != 'auto' else 'cuda')


def torch_load(path):
    # Only load our own trusted training checkpoints (optimizer/RNG states included).
    return torch.load(path, map_location='cpu', weights_only=False)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def data_signature(c):
    return {k: file_hash(c['data'][k]) if c['data'].get(k) else None for k in PATH_KEYS}


def run_signature(c):
    v = copy.deepcopy(c)
    for key in ('_config_path', 'output_dir', 'device', 'num_workers', 'threads'):
        v.pop(key, None)
    # More epochs are allowed on resume; all optimizer/batch settings stay fixed.
    for stage in ('stage1', 'stage2'):
        v[stage].pop('epochs', None)
    v['data'] = data_signature(c)
    return digest(v)
