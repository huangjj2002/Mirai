"""DCMTK conversion following the Mirai README, with explicit path mapping.

This prepares candidate images only: no cancer labels, splits or four-view selection.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile

FLAGS = ['+on2', '--min-max-window']
PRESENTATION_SOP = '1.2.840.10008.5.1.4.1.1.1.2'
MANIFEST_FIELDS = ['patient_id', 'exam_id', 'laterality', 'view', 'file_path',
                   'source_dicom_path', 'resolved_dicom_path', 'device_model']


def resolve_dicom(raw, root=None, strip_prefix=None):
    if not raw.strip():
        raise ValueError('Empty source DICOM path')
    raw = raw.strip().replace('\\', '/')
    source = PurePosixPath(raw)
    if strip_prefix:
        if not root:
            raise ValueError('--strip-prefix requires --dicom-root')
        try:
            source = source.relative_to(PurePosixPath(strip_prefix.replace('\\', '/')))
        except ValueError:
            raise ValueError('Source does not match --strip-prefix')
    if source.is_absolute():
        path = Path(str(source)).resolve()
    else:
        if not root:
            raise ValueError('Relative DICOM path requires --dicom-root')
        base = Path(root).expanduser().resolve()
        path = (base / str(source)).resolve()
        if not path.is_relative_to(base):
            raise ValueError('DICOM path escapes --dicom-root')
    if not path.is_file():
        raise FileNotFoundError(f'DICOM not found: {path}')
    return path


def candidate_reason(row):
    if row['FinalImageType'].strip() != '2D':
        return 'not_2D'
    if row['ImageLateralityFinal'].strip() not in ('L', 'R') or row['ViewPosition'].strip() not in ('CC', 'MLO'):
        return 'not_standard_view'
    if row['spot_mag'].strip().casefold() not in ('', '0', '0.0', 'false'):
        return 'spot_mag_or_unknown'
    if not row['empi_anon'].strip() or not row['acc_anon'].strip():
        return 'missing_identity'
    return None


def check_header(ds, row):
    intent = str(ds.get('PresentationIntentType', '')).strip().upper()
    sop = str(ds.get('SOPClassUID', ''))
    if intent == 'FOR PROCESSING' or (intent != 'FOR PRESENTATION' and sop != PRESENTATION_SOP):
        raise ValueError('Require presentation DICOM; raw processing images are not interchangeable')
    if int(ds.get('NumberOfFrames', 1)) != 1:
        raise ValueError('Multi-frame DICOM is excluded')
    if int(ds.get('SamplesPerPixel', 0)) != 1 or str(ds.get('PhotometricInterpretation', '')) not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('Require monochrome mammogram')
    for tag, field in [('ImageLaterality', 'ImageLateralityFinal'), ('ViewPosition', 'ViewPosition')]:
        value = str(ds.get(tag, '')).strip()
        if value and value != row[field].strip():
            raise ValueError(f'DICOM {tag} conflicts with CSV')
    width, height = int(ds.get('Columns', 0)), int(ds.get('Rows', 0))
    if min(width, height) <= 0:
        raise ValueError('Invalid DICOM dimensions')
    return width, height


def verify_png(path, expected_size):
    from PIL import Image
    with path.open('rb') as stream:
        head = stream.read(29)
    if len(head) != 29 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        raise ValueError('Invalid PNG header')
    width, height, depth, color = struct.unpack('>IIBB', head[16:26])
    if depth != 16 or color != 0 or (width, height) != expected_size:
        raise ValueError('Expected original-size 16-bit grayscale PNG')
    with Image.open(path) as image:
        image.load()
        lo, hi = image.getextrema()
    if lo == hi:
        raise ValueError('Constant-valued image')
    return {'width': width, 'height': height, 'min': lo, 'max': hi}


def render_one(source, destination, size, executable, version):
    stat = source.stat()
    provenance = {'source': str(source), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                  'dcmtk_version': version, 'flags': FLAGS, 'size': list(size)}
    sidecar = destination.with_suffix('.json')
    if destination.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text(encoding='utf-8'))['provenance'] != provenance:
            raise ValueError('Existing PNG has missing/different conversion provenance; use a new output directory')
        verify_png(destination, size)
        return 'reused'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix='.png', dir=destination.parent)
    os.close(fd)
    temp = Path(name)
    try:
        result = subprocess.run([executable, *FLAGS, str(source), str(temp)],
                                capture_output=True, text=True, errors='replace', timeout=180)
        if result.returncode:
            raise RuntimeError(f'DCMTK exited {result.returncode}: {result.stderr[-1500:]}')
        properties = verify_png(temp, size)
        temp.replace(destination)
        sidecar.write_text(json.dumps({'provenance': provenance, 'png': properties,
            'dcmtk_stderr': result.stderr[-2000:]}, indent=2), encoding='utf-8')
        return 'converted'
    finally:
        temp.unlink(missing_ok=True)


def convert_table(args):
    import pydicom
    if args.limit < 0:
        raise ValueError('--limit must be nonnegative')
    metadata = Path(args.image_metadata_csv).expanduser().resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if args.strip_prefix and not args.dicom_root:
        raise ValueError('--strip-prefix requires --dicom-root')
    executable, version = None, None
    if not args.dry_run:
        executable = shutil.which(args.dcmtk)
        if not executable:
            raise RuntimeError('dcmj2pnm is unavailable; install DCMTK with PNG support or pass --dcmtk')
        check = subprocess.run([executable, '--version'], capture_output=True, text=True, errors='replace', timeout=20)
        if check.returncode:
            raise RuntimeError(f'Cannot run DCMTK: {check.stderr}')
        version = check.stdout.strip() + '\n' + check.stderr.strip()
    out = Path(args.output_dir).expanduser().resolve()
    if metadata in [out/'images.csv', out/'errors.csv', out/'conversion.json']:
        raise ValueError('Output would overwrite the metadata input')
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    scan_complete = True
    csv.field_size_limit(16 * 1024 * 1024)
    with metadata.open(encoding='utf-8-sig', newline='') as stream, \
            (out/'images.csv').open('w', encoding='utf-8', newline='') as success, \
            (out/'errors.csv').open('w', encoding='utf-8', newline='') as failures:
        reader = csv.DictReader(stream)
        needed = {'empi_anon', 'acc_anon', 'FinalImageType', 'ViewPosition',
                  'ImageLateralityFinal', 'spot_mag', args.dicom_column}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'Metadata missing columns: {sorted(needed-set(reader.fieldnames or []))}')
        writer = csv.DictWriter(success, fieldnames=MANIFEST_FIELDS)
        errors = csv.DictWriter(failures, fieldnames=['csv_line', 'source_dicom_path', 'error'])
        writer.writeheader(); errors.writeheader()
        for row in reader:
            counts['rows_scanned'] += 1
            if None in row or any(row.get(k) is None for k in needed):
                raise ValueError(f'Malformed CSV line {reader.line_num}')
            reason = candidate_reason(row)
            if reason:
                counts[reason] += 1
                continue
            if args.limit and counts['attempted'] >= args.limit:
                scan_complete = False
                break
            counts['attempted'] += 1
            raw = row[args.dicom_column].strip()
            try:
                source = resolve_dicom(raw, args.dicom_root, args.strip_prefix)
                ds = pydicom.dcmread(source, stop_before_pixels=True)
                size = check_header(ds, row)
                model = str(ds.get('ManufacturerModelName', '')).strip()
                reported = row.get('ManufacturerModelName', '').strip()
                if model and reported and model != reported:
                    raise ValueError('DICOM device model conflicts with CSV')
                if args.dry_run:
                    counts['headers_ok'] += 1
                    continue
                token = hashlib.sha256(str(source).encode()).hexdigest()
                destination = out/'png'/token[:2]/(token+'.png')
                status = render_one(source, destination, size, executable, version)
                counts[status] += 1
                writer.writerow(dict(zip(MANIFEST_FIELDS, [
                    row['empi_anon'].strip(), row['acc_anon'].strip(),
                    row['ImageLateralityFinal'].strip(), row['ViewPosition'].strip(),
                    str(destination), row.get('anon_dicom_path', '').strip(), str(source), model or reported])))
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, pydicom.errors.InvalidDicomError) as exc:
                counts['errors'] += 1
                errors.writerow({'csv_line': reader.line_num, 'source_dicom_path': raw, 'error': str(exc)})
            if counts['attempted'] % 100 == 0:
                print(dict(counts), flush=True)
    report = {'counts': dict(counts), 'scan_complete': scan_complete,
              'dry_run': args.dry_run, 'settings': vars(args), 'dcmtk_version': version,
              'recipe': FLAGS, 'extra_flip_crop_resize': False,
              'cohort_and_outcomes_validated': False}
    (out/'conversion.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'counts': dict(counts), 'output': str(out), 'scan_complete': scan_complete}, indent=2))
    if counts['errors'] or not counts['attempted']:
        raise RuntimeError('Conversion/header audit incomplete; see errors.csv and conversion.json')
