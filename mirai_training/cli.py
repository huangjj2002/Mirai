import argparse
import json
import sys
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description='Original-structure Mirai: data audit, two-stage training and evaluation')
    sub=p.add_subparsers(dest='command')
    fields=sub.add_parser('audit-fields', help='Inspect raw EMBED CSV fields without images or PyTorch')
    fields.add_argument('--clinical-csv', required=True)
    fields.add_argument('--image-metadata-csv', required=True)
    fields.add_argument('--output-dir', default='outputs/embed_fields')
    fields.add_argument('--max-rows', type=int, default=0, help='Per table; 0 scans all rows, positive values are a preview only')
    fields.add_argument('--encoding', default='utf-8-sig')
    convert=sub.add_parser('convert-dicom', help='Presentation DICOM to PNG16 using the Mirai DCMTK recipe')
    convert.add_argument('--config', help='Conversion JSON; explicitly supplied CLI values take precedence')
    convert.add_argument('--image-metadata-csv')
    convert.add_argument('--output-dir')
    convert.add_argument('--dicom-column')
    convert.add_argument('--dicom-root', help='Root for relative paths, or replacement root with --strip-prefix')
    convert.add_argument('--strip-prefix', help='Explicit obsolete prefix to remove from CSV paths')
    convert.add_argument('--dcmtk')
    convert.add_argument('--limit', type=int, help='Maximum candidate images attempted; default 16, 0 means all')
    convert.add_argument('--dry-run', action='store_true', help='Check DICOM headers/paths without conversion')
    for command in ('check-data','train','extract-features','evaluate','smoke-test'):
        s=sub.add_parser(command)
        s.add_argument('--config',default=str(Path(__file__).resolve().parent.parent/'configs/mirai.yaml'))
        s.add_argument('--device',default=None,help='cpu, cuda, cuda:0 (overrides config)')
        if command=='check-data':
            s.add_argument('--pixels',action='store_true',help='Verify PNG16 headers and integrity')
            s.add_argument('--compute-stats',action='store_true',help='Recompute TRAIN normalization')
        elif command=='train':
            s.add_argument('--stage',choices=['1','2','all'],default='all')
            s.add_argument('--resume',help='Our own last.pt/best.pt checkpoint')
        elif command=='extract-features':
            s.add_argument('--encoder',help='Stage1 checkpoint; defaults to this run stage1/best.pt')
        elif command=='evaluate':
            s.add_argument('--checkpoint',required=True)
            s.add_argument('--split',choices=['dev','test'],default='test')
        else:
            s.add_argument('--full-resolution',action='store_true',help='Also forward one original-size PNG through encoder')
    return p


def conversion_args(args):
    defaults = {'image_metadata_csv': None, 'output_dir': None,
                'dicom_column': 'anon_dicom_path', 'dicom_root': None,
                'strip_prefix': None, 'dcmtk': 'dcmj2pnm', 'limit': 16}
    if args.config:
        path = Path(args.config).expanduser().resolve()
        config = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(config, dict) or set(config) - set(defaults):
            raise ValueError('Conversion config must contain only supported conversion options')
        for key in ('image_metadata_csv', 'output_dir', 'dicom_root'):
            if config.get(key):
                config[key] = str((path.parent / Path(config[key]).expanduser()).resolve())
        defaults.update(config)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not args.image_metadata_csv or not args.output_dir:
        raise ValueError('Conversion requires image_metadata_csv and output_dir (CLI or config)')
    if isinstance(args.limit, bool) or not isinstance(args.limit, int) or args.limit < 0:
        raise ValueError('Conversion limit must be a nonnegative integer')
    return args


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if not args.command:
        p.print_help(); return 0
    try:
        if args.command=='audit-fields':
            from .field_audit import run_audit
            run_audit(args.clinical_csv, args.image_metadata_csv, args.output_dir,
                      args.max_rows, args.encoding)
            return 0
        if args.command=='convert-dicom':
            args=conversion_args(args)
            from .dicom_conversion import convert_table
            convert_table(args)
            return 0
        from .config import load_config, torch_load
        from .engine import check_data, evaluate, extract_features, train_stage
        c=load_config(args.config)
        if args.device:
            c['device']=args.device
        if args.command=='check-data':
            check_data(c,args.pixels,args.compute_stats)
        elif args.command=='extract-features':
            extract_features(c,args.encoder)
        elif args.command=='evaluate':
            evaluate(c,args.checkpoint,args.split)
        elif args.command=='smoke-test':
            from .smoke import run_smoke
            run_smoke(c,args.full_resolution,args.device or 'cpu')
        elif args.stage!='all':
            train_stage(c,int(args.stage),args.resume)
        else:
            resume_stage=torch_load(args.resume)['stage'] if args.resume else 1
            if resume_stage==1:
                train_stage(c,1,args.resume)
                extract_features(c)
                train_stage(c,2)
            elif resume_stage==2:
                train_stage(c,2,args.resume)
            else:
                raise ValueError('Invalid checkpoint stage')
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,ImportError) as e:
        print(f'ERROR: {e}',file=sys.stderr)
        return 2
