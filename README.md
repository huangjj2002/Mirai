# Mirai on EMBED

用于 EMBED 的 Mirai 两阶段训练与数据准备代码。统一入口为根目录 `main.py`。

保留官方图像编码器、四视图聚合、34 个风险因素预测头、五年累积风险输出及设备判别器。缺失监督使用掩码；不下载 Mirai 风险模型权重，第一阶段可使用 ImageNet 初始化。

完整使用说明见 [README_Mirai.md](README_Mirai.md)。项目尚未完成服务器真实影像转换核查及完整训练，工程测试通过不代表论文性能已复现。

## 仓库内容

- `mirai_training/`：模型、数据加载、训练、评估、DICOM 转换及测试。
- `configs/`：训练配置及通用转换配置示例。
- `analysis/mirai_embed/adapt_embed.py`：显式结局与四视图图像表导出。
- `data/mirai/*.example.*`：虚构记录的格式示例。

这是独立的 Mirai 项目目录。原始 EMBED 表、DICOM、PNG、患者级审计文件、模型权重和运行输出均不上传。根目录使用白名单式 `.gitignore`；新增需要跟踪的根文件或目录时，须同步更新规则。

## 在服务器使用

克隆仓库后，进入项目根目录。训练依赖与 CUDA 环境按完整说明配置。查看可用命令：

```bash
python main.py --help
```

从 DICOM 准备图像时，安装 `requirements-conversion.txt` 和支持 PNG 的 DCMTK，然后复制通用配置：

```bash
cp configs/embed_conversion.example.json configs/embed_conversion_server.json
```

将新文件中的表路径、旧前缀及当前 DICOM 根目录改为服务器实际值，再运行：

```bash
python main.py convert-dicom --config configs/embed_conversion_server.json
```

本机专用的 `embed_conversion_server.json` 被 Git 忽略。默认尝试 16 张，检查通过后才用 `--limit 0` 扩展转换。转换清单仍需完成每视图唯一选择、结局确认及患者级数据分区后才能训练。

`test_adapter.py` 的原版加载器对照测试还需要另外取得官方源码，置于 `code/Mirai-master/Mirai-master`；该第三方完整仓库不包含在本仓库内。

## 200 位患者试跑

安装训练依赖、转换依赖、`requirements-data.txt` 和 DCMTK。此流程使用真实图像，选择 200 位不同患者、每人一次检查、每次恰好一张 L/R CC/MLO，共 800 张，按患者分为 160/20/20。模型结构与原分辨率不变，两阶段先各跑 1 个 epoch。

试跑有意抽取 50 例五年内记录到癌症的候选和 150 例其他候选，方便检查两类监督及指标流程。该比例不是 EMBED 发病率，结果不得作为正式模型性能或校准结论。`selection.json` 会报告训练集是否至少包含两个原模型识别的设备类别；不足时不能完成原设备对抗训练，需重新选取队列。

```bash
python main.py prepare-pilot \
  --clinical-csv /opt/NAS3/datasets/internal/EMBED/tables/magview_all_cohorts_anon_HITI.csv \
  --image-metadata-csv /opt/NAS3/datasets/internal/EMBED/tables/metadata_all_cohort_with_ROI_HITI.csv \
  --dicom-root /opt/NAS3/datasets/internal/EMBED/anon_dicom \
  --strip-prefix /mnt/NAS2/mammo/anon_dicom \
  --patients 200 --positive-candidates 50 \
  --label-policy recorded-screening --output-dir outputs/pilot200

python main.py convert-dicom --config outputs/pilot200/conversion.json

python main.py finalize-pilot --pilot-dir outputs/pilot200 \
  --outcomes-csv outputs/pilot200/outcomes_pilot.csv

python main.py check-data --config outputs/pilot200/train.yaml --pixels --compute-stats
python main.py train --config outputs/pilot200/train.yaml --stage all
python main.py evaluate --config outputs/pilot200/train.yaml \
  --checkpoint outputs/pilot200/training/stage2/best.pt --split test
```

按顺序执行，遇到错误先停止，不要继续训练。新队列目录必须为空，避免覆盖既有患者划分。图像表分块读取，筛选时核对患者/检查身份、临床与采集日期、完整视图及文件存在性；多张同视图时排除该检查，不随意选择一张。选择步骤仍需扫描全量表，转换步骤才限制为这 800 张。

`--label-policy recorded-screening` 是显式启用的临时研究规则：

- 乳腺癌事件包括 `path_severity=0/1`，以患者首次有效 `procdate_anon` 为事件日期，必须晚于基线。取材早于关联检查、报告早于取材、相关日期缺失或间隔超过 365 天的癌症记录会使该患者退出候选集。
- 无五年内癌症事件的候选必须有至少五年后明确 N/B 阴性筛查记录，以该记录日期作为观察截止候选。缺少病理本身不构成阴性；已知患癌 K 却没有可确定日期的癌症事件也会排除。
- 基线采用临床与影像日期一致的女性筛查检查。每位患者选最早满足所有筛选条件的检查，排除基线前或同日已有记录的癌症。
- 此规则依赖“院内记录捕获相关事件”的假设，无法保证院外患癌事件、既往癌症史和连续随访均被完整记录。正式实验必须复核事件、随访及入组定义。

默认不加该策略时，只生成 `outcomes_to_review.csv`，其中事件/随访标签留空、候选日期另列；不会直接生成训练标签。有已确认结局表时，可将它交给 `finalize-pilot --outcomes-csv`，但需要保留所选患者、检查、基线日期及分区。

`check-data` 若排除了任何样本，应先查 `training/data_audit.json`，不能把少于 200 例的结果当成完整 200 例运行。正式训练仍使用基础配置；本流程生成的 `train.yaml` 仅用于试跑。

## 上游来源

模型代码来自 [yala/Mirai](https://github.com/yala/Mirai)，固定提交 `3904a9eaca046378a194b1eb8c62fa32f45ce83b`。所需源码、原始 MIT 许可证和校验清单保留在 `mirai_training/vendor/`。上游代码许可不表示本仓库包含 EMBED 数据或提供其使用授权。
