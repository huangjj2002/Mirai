"""Stream raw EMBED tables using the standard library; emit aggregate field evidence.

Names are discovery hints, not permission to turn a field into a model feature.
This audit does not build outcomes, join patient records, or change training.
"""
import csv
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

UNKNOWN = {'unknown', 'unavailable', 'unreported', 'not recorded', 'patient declines',
           'unknown, unavailable or unreported', 'unreported, unknown, unavailable',
           'nan', 'null', 'none', 'n/a', 'na'}
CATEGORIES = {
    'race_desc', 'ethnicity_desc', 'ethnic_group_desc', 'gender_desc', 'patientsex',
    'tissueden', 'vtype', 'type', 'asses', 'path_severity', 'path_group',
    'manufacturer', 'manufacturermodelname', 'finalimagetype', 'imagelateralityfinal',
    'viewposition', 'spot_mag', 'cohort_num', 'photometricinterpretation',
    'presentationintenttype', 'bitsstored', 'rows', 'columns',
    'patientage', 'png_flipped', 'ssc_roi_flipped', 'has_pix_array',
}
for _i in range(1, 11):
    CATEGORIES.add(f'path{_i}')

CLINICAL_HINTS = re.compile(
    r'family|fam_hist|fhx|brca|genetic|menarch|menopaus|pregnan|parous|parity|'
    r'gravida|hormone|(^|_)hrt|weight|height|bmi|ovarian|ashkenazi|'
    r'biopsy|lcis|hyperplasia|prior_hist|follow.?up|death|cancer_date|last_contact', re.I)

# The list is deliberately descriptive. No crosswalk is applied by this tool.
FACTOR_HINTS = {
    'age': ('age_at_study', 'PatientAge'), 'density': ('tissueden',),
    'race': ('RACE_DESC', 'ETHNICITY_DESC'),
    'binary_family_history': ('family_history', 'family_hx', 'fhx'),
    'binary_biopsy_benign': ('biopsy_hyperplasia', 'benign_biopsy'),
    'binary_biopsy_LCIS': ('biopsy_LCIS', 'LCIS'),
    'binary_biopsy_atypical_hyperplasia': ('biopsy_atypical_hyperplasia', 'atypical_hyperplasia'),
    'menarche_age': ('menarche_age', 'age_at_menarche'),
    'menopause_age': ('menopause_age', 'age_at_menopause'),
    'first_pregnancy_age': ('first_pregnancy_age', 'age_at_first_birth'),
    'menopausal_status': ('menopausal_status',), 'parous': ('parous', 'parity'),
    'weight': ('weight', 'PatientWeight'), 'height': ('height', 'PatientSize'),
    'brca': ('brca', 'brca1', 'brca2'), 'ashkenazi': ('ashkenazi',),
    'ovarian_cancer': ('ovarian_cancer',),
    'hrt_type': ('hrt_type',), 'hrt_duration': ('hrt_duration',),
    'hrt_years_ago_stopped': ('hrt_years_ago_stopped',),
}
HISTORY = {'binary_biopsy_benign', 'binary_biopsy_LCIS', 'binary_biopsy_atypical_hyperplasia'}


def scan_table(path, max_rows=0, encoding='utf-8-sig'):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    counts, sentinel_counts, categories = [], [], {}
    n = 0
    truncated = False
    csv.field_size_limit(16 * 1024 * 1024)
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    with opener(path, 'rt', encoding=encoding, newline='') as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f'Empty CSV: {path}')
        if len(set(header)) != len(header):
            raise ValueError(f'Duplicate column names in {path}; inspect the source header first')
        counts = [0] * len(header)
        sentinel_counts = [0] * len(header)
        categories = {i: Counter() for i, col in enumerate(header) if col.casefold() in CATEGORIES}
        category_overflow = Counter()
        for row in reader:
            if not row:
                continue
            if max_rows and n >= max_rows:
                truncated = True
                break
            if len(row) != len(header):
                raise ValueError(f'{path.name}: CSV line {reader.line_num} has {len(row)} values, expected {len(header)}')
            for i, value in enumerate(row):
                value = value.strip()
                counts[i] += bool(value)
                sentinel_counts[i] += value.casefold() in UNKNOWN
                if i in categories and value:
                    # Bound memory and avoid publishing free-text fields or record IDs.
                    values = categories[i]
                    if value in values or len(values) < 100:
                        values[value] += 1
                    else:
                        category_overflow[i] += 1
            n += 1
            if n % 250000 == 0:
                print(f'{path.name}: {n:,} rows inspected', flush=True)
    fields = []
    for i, name in enumerate(header):
        item = {'name': name, 'nonblank': counts[i], 'blank': n - counts[i],
                'nonblank_percent': round(100 * counts[i] / n, 3) if n else None,
                'explicit_unknown_tokens': sentinel_counts[i]}
        if i in categories:
            item['categories'] = dict(categories[i].most_common())
            item['other_category_rows'] = category_overflow[i]
        fields.append(item)
    after = path.stat()
    if stat.st_size != after.st_size or stat.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f'Source changed while reading: {path}')
    print(f'{path.name}: {n:,} rows, {len(header)} columns; ' + ('PREVIEW' if truncated else 'complete'), flush=True)
    return {'path': str(path), 'bytes': stat.st_size, 'rows_scanned': n,
            'column_count': len(header), 'complete': not truncated, 'fields': fields}


def find_fields(tables, names):
    lower = {s.casefold() for s in names}
    return [f'{key}.{f["name"]}' for key, table in tables.items()
            for f in table['fields'] if f['name'].casefold() in lower]


def factor_mapping(tables):
    vendor = Path(__file__).parent / 'vendor/mirai_base.json'
    original = json.loads(vendor.read_text(encoding='utf-8'))
    keys = original['search_space']['risk_factor_keys'][0].split()
    out = []
    for key in keys:
        hits = find_fields(tables, (key,) + FACTOR_HINTS.get(key, ()))
        status = '发现同名/别名候选；需核对定义、时间及编码'
        note = '列名匹配不等于可直接接入。'
        if key in ('age', 'density'):
            note = '现有加载器支持 age_at_study/tissueden；按患者+检查核对冲突，密度仅接受 1–4。'
            if key == 'age':
                note += ' PatientAge 仅作待核对候选，不能直接按浮点年龄读取或自动覆盖临床年龄。'
        elif key == 'race':
            note = '核对 RACE_DESC/ETHNICITY_DESC 取值及官方种族编码；ETHNIC_GROUP_DESC 不可直接替代 race。'
        elif key in ('height', 'weight'):
            note = '必须核对单位；不能直接将 DICOM PatientSize/PatientWeight 的原始值套用 Mirai 身高/体重分箱。'
        if not hits:
            status = '未找到所列候选列名'
            note = '不能断言服务器没有该信息；还应检查报告末尾的全部表头、相关候选列及数据字典。'
        if key in HISTORY:
            path_hits = find_fields(tables, ('path1', 'procdate_anon', 'pdate_anon'))
            if path_hits:
                hits += path_hits
                status += '；另有病理历史推导线索'
                note = '仅可用基线之前且当时已知的病理记录；需核对全部 path1…path10 及原模型定义，未记录不能当作无既往史。'
        out.append({'factor': key, 'candidate_columns': list(dict.fromkeys(hits)),
                    'status': status, 'note': note})
    return out


def escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def write_report(result, path):
    tables = result['tables']
    complete = all(t['complete'] for t in tables.values())
    lines = ['# EMBED 原始字段核查', '', f"核查时间（UTC）：{result['checked_at_utc']}", '',
             '范围：' + ('全表扫描。' if complete else '包含限行预览，覆盖率仅代表已扫描行，不能外推全量。'), '',
             '只读取 CSV；未读取影像、训练模型或生成癌症/随访标签。统计以原始行计，临床行不等于患者或检查。', '',
             '| 表 | 扫描行数 | 列数 | 完整扫描 |', '|---|---:|---:|---|']
    for name, table in tables.items():
        lines.append(f'| {name} | {table["rows_scanned"]:,} | {table["column_count"]} | {table["complete"]} |')
    lines += ['', '## Mirai 原始风险因素对应', '',
              '保留原模型全部风险因素头。下表仅列字段线索，未执行类别映射；不存在或未知的监督应使用 mask。', '',
              '| 原模型因素 | 本次发现的候选列 | 判断与注意事项 |', '|---|---|---|']
    for f in result['mirai_factors']:
        lines.append(f'| `{f["factor"]}` | {escape(", ".join(f["candidate_columns"]) or "无精确候选")} | {escape(f["status"] + "。" + f["note"])} |')
    lines += ['', '## 标签、设备与影像处理', '',
              '- 患者与检查关联：优先检查 empi_anon、acc_anon，不能仅按检查 ID 或 CSV 行索引连接；本命令未校验实体关联。',
              '- 结局候选：path_severity、path1…path10、bside、procdate_anon、pdate_anon。病理可能回填到较早的影像行，必须按真实事件时间整理。',
              '- 缺少病理不代表阴性；最近一条就诊记录也不自动等于连续无癌随访。需要明确结局与删失规则。',
              '- 设备：Manufacturer、ManufacturerModelName、FinalImageType。未知型号不要硬塞进 Mirai 已有四类；这一步只列取值供核对。',
              '- 四视图：ImageLateralityFinal、ViewPosition、FinalImageType、spot_mag；anon_dicom_path/png_path 等定位图像。列存在不代表影像文件已下载。',
              '- 灰阶与尺寸：Rows、Columns、BitsStored、PhotometricInterpretation、WindowCenter/Width、PixelSpacing 等，用于预处理与质量核查，不直接增加模型结构。',
              '- 内部版可有 png_path、png_path_cropped、PNG_flipped、PNG_ROI_coords、DCM_ROI_coords、num_roi。路径存在不等于文件可读；整图与裁剪图、PNG 与 DICOM 坐标不可混用，不能直接按表头判定翻转规则。',
              '- 空白率与有效率不同：Unknown 等显式未知、日期异常、密度 5、默认 0 都需按字段进一步核对。', '',
              '## 额外临床字段线索', '',
              '按列名关键词发现，下列内容仍需字典解释。遗漏不代表没有信息：', '']
    lines += [f'- `{escape(s)}`' for s in result['additional_clinical_candidates']] or ['未发现匹配关键词的列。']
    for name, table in tables.items():
        lines += ['', f'## {name} 全部字段与覆盖率', '', f'输入：`{table["path"]}`', '',
                  '| 字段 | 非空行数 | 非空比例 | 显式未知行数 |', '|---|---:|---:|---:|']
        for f in table['fields']:
            pct = f'{f["nonblank_percent"]:.3f}%' if f['nonblank_percent'] is not None else '无数据'
            lines.append(f'| `{escape(f["name"] or "<空表头/导出索引>")}` | {f["nonblank"]:,} | {pct} | {f["explicit_unknown_tokens"]:,} |')
        lines += ['', f'## {name} 关键类别取值', '']
        for f in table['fields']:
            if 'categories' not in f:
                continue
            lines += [f'### `{escape(f["name"])}`', '', '| 值 | 行数 |', '|---|---:|']
            lines += [f'| {escape(k)} | {v:,} |' for k, v in f['categories'].items()]
            if f['other_category_rows']:
                lines.append(f'| 其余类别（限存前 100 类） | {f["other_category_rows"]:,} |')
            lines.append('')
    lines += ['', '## 定义查找来源', '',
              '- [EMBED 官方字段字典](https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/resources/AWS_Open_Data_Clinical_Legend.csv)',
              '- [数据表结构](https://docs.hitilab.com/docs/datasets/embed/structure)',
              '- [标签定义](https://docs.hitilab.com/docs/datasets/embed/label-assignment)',
              '- 原始 Mirai 风险因素顺序来自项目 vendor/mirai_base.json；本报告未修改训练配置。', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def run_audit(clinical_csv, image_metadata_csv, output_dir, max_rows=0, encoding='utf-8-sig'):
    if max_rows < 0:
        raise ValueError('--max-rows must be nonnegative')
    # Resolve both sources before reading either potentially very large file.
    inputs = [Path(clinical_csv).expanduser().resolve(), Path(image_metadata_csv).expanduser().resolve()]
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(f'CSV not found on THIS machine: {p}')
    out = Path(output_dir).expanduser().resolve()
    outputs = [out / 'field_audit.json', out / 'report.md']
    if any(p in inputs for p in outputs):
        raise ValueError('Output paths would overwrite an input CSV')
    tables = {'clinical': scan_table(inputs[0], max_rows, encoding),
              'metadata': scan_table(inputs[1], max_rows, encoding)}
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'max_rows_per_table': max_rows, 'tables': tables,
              'mirai_factors': factor_mapping(tables),
              'additional_clinical_candidates': [f'{name}.{f["name"]}' for name, table in tables.items()
                  for f in table['fields'] if CLINICAL_HINTS.search(f['name'])]}
    out.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(result, outputs[1])
    print(f'Report: {outputs[1]}\nAggregate evidence: {outputs[0]}', flush=True)
    return result
