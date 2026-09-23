# Mirai：原版结构、两阶段重新训练

入口在工作区根目录 `main.py`。实现不改动 `code/Mirai-master/Mirai-master`，不加载任何 Mirai 预训练权重。正式第一阶段仅使用原版指定的 ImageNet ResNet-18 初始化；第二阶段使用自己训练出的编码器。

## 1. 模型结构与源码对应

`mirai_training/vendor/onconet` 保存官方提交 `3904a9eaca046378a194b1eb8c62fa32f45ce83b` 的必要源码和 MIT 许可。20 个源码文件只更换导入命名空间；设备常量改为从独立数据常量模块导入，类别及编号不变。模型类的方法体未重写。`vendor/manifest.json` 记录源文件和副本指纹，验证会检测未经记录的变化。

| 模块 | 原版类/结构 |
|---|---|
| 单视图编码器 | `CustomResnet` → `ResNet`，BasicBlock `[2,2,2,2]`，全局最大池化，512 维 |
| 四视图聚合 | `AllImageTransformer`，保留投影、体位/侧别/时间嵌入、自定义注意力与前馈层 |
| 风险因素 | `RiskFactorPool`，34 个头、100 维完整拼接；原版顺序、类别和激活函数 |
| 多年风险 | `Cumulative_Probability_Layer`，基线 logit 加非负年度增量，sigmoid 得到五年输出 |
| 设备对抗 | `Discriminator`，两个带 BatchNorm/ReLU 的隐藏层，四类输出 |

默认验证配置为 512 维、8 头、注意力池化。其他原配置候选为 1024 维、16 头以及最大/平均池化；参数文件保留在 `vendor/mirai_full.json`。默认值是公开搜索空间中的一个配置，不是已经核实的作者最终最佳配置。

正式代码直接调用这些原版类；外层训练器负责缺失标签、数据、优化器、缓存与日志。风险因素池只绑定一个**无参数的辅助损失方法**，利用额外有效性掩码屏蔽缺失监督；不添加层、不删预测头、不改变 forward。全部标签有效时，结构、输出、辅助损失及梯度与原版比较。

## 2. 环境

### 先检查服务器 EMBED 原始字段

此命令只依赖 Python 标准库，不需要 PyTorch、pandas、GPU 或影像文件。逐行读取两张原始表，输出全部列名、非空率、关键类别分布、34 个原版风险因素的候选字段。不会改动原表、训练配置或构造结局。

将更新后的项目同步到服务器，在含 `main.py` 的项目根目录运行：

```bash
python main.py audit-fields \
  --clinical-csv /opt/NAS3/datasets/internal/EMBED/tables/magview_all_cohorts_anon_HITI.csv \
  --image-metadata-csv /opt/NAS3/datasets/internal/EMBED/tables/metadata_all_cohort_with_ROI_HITI.csv \
  --output-dir outputs/embed_fields_server
```

文件名下划线前不要加反斜线。默认扫描完整 CSV；每 25 万行显示一次进度。`--max-rows 50000` 可快速预览两张表的前 5 万行，但预览缺失率不代表全表。每次输出会覆盖同一输出目录内的报告，预览请使用不同输出目录。

结果：`outputs/embed_fields_server/report.md` 是可读报告，`field_audit.json` 是汇总证据。两者不导出患者 ID、逐行记录或影像路径列表；保留源文件路径、列名和关键类别汇总。发回这两个小文件即可继续核对实际字段，不需要传回原始 CSV。

字段名只用于发现候选，不等于可直接映射：尤其是种族、家族史、身高/体重单位、既往活检和随访时间。服务器文件尚未实测；不能由 `all_cohorts` 文件名推定数据完整性。

目标服务器使用 Python 3.11、独立环境。依赖清单为 `requirements-mirai.txt`；不需要安装原仓库全部历史依赖，也不依赖 Git 仓库状态或 Linux `pwd` 模块。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-mirai.txt
python main.py --help
```

以上是 CUDA 12.8 构建的服务器安装方案，服务器驱动与显卡兼容性须在实机检查。本次本地测试使用现有 Python 3.10.0 / torch 1.13.1+cu117 的 **CPU**，未升级全局环境，也未验证 GPU 训练。当前旧 torch 不支持本机 sm_120，选择 GPU 时入口会提前报错，不会因为 `cuda.is_available()` 为 True 就继续。

本地验证的其余依赖：numpy 1.26.4、scipy 1.11.4、scikit-learn 1.0.2、Pillow 9.5.0、PyYAML 6.0.3。服务器锁定版本尚未在服务器安装验证。CUDA 安装命令对应 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v271)。

## 3. 数据接口

配置文件为 `configs/mirai.yaml`，配置中的相对路径相对于**配置文件目录**；影像 `file_path` 必须是运行机器上的绝对路径。默认真实输入 `data/mirai/metadata.csv` 尚不存在，不能将示例文件或候选清单当作正式队列。

### 已准备的 PNG16 与结局 CSV

参考 `data/mirai/metadata.example.csv`，每张图像一行，八个基本字段：

`patient_id, exam_id, laterality, view, file_path, years_to_cancer, years_to_last_followup, split_group`

- 每次检查恰好 R-CC、R-MLO、L-CC、L-MLO 四张。重复视图必须显式选择，程序不会随机取第一张。
- ID 始终读取为字符串；同一患者只能属于一个 train/dev/test 分区。
- `years_to_cancer` 是原版零起始事件年索引：0 为第一年、2 为第三年；没有已观察事件时用 100。`years_to_last_followup` 是可靠随访的完整年数。
- 风险标签与 mask 延续原版：第三年事件为 y `[0,0,1,1,1]`、mask `[1,1,1,0,0]`；无事件、两年随访为 y 全零、mask `[1,1,0,0,0]`。不能将未知未来填成阴性。
- 本次不自动从原始临床表推导首次癌症、阴性随访或分区。可继续使用已有 `analysis/mirai_embed/adapt_embed.py export`，但前提是已准备审定结局表与选定影像清单。
- 需事先将 presentation DICOM 转为 PNG16，可使用下面的 `convert-dicom`。训练加载器不直接接受 DICOM 或 PNG8。

输入缩放沿用 PIL I 模式、双线性插值、原版像素和判断翻转；训练阶段加入垂直翻转和 ±20 度整数旋转。灰度展开为三通道，尺寸为 width=1664、height=2048。归一化由训练图像计算；明确提供配置 mean/std 时使用该值并记录来源。原版在 NumPy 1.x Windows 上用 int32 累加像素会溢出；此处使用宽精度求和，已经逐像素对照原版 Linux int64 行为，避免错误翻转。

### 设备标签

CSV 可增加 `device_model`，或增加 `source_dicom_path` 并在配置中提供 EMBED `image_metadata_csv`，精确关联 `anon_dicom_path` 与 `ManufacturerModelName`。不会从 PNG 文件名猜测源 DICOM。两种来源不一致时报错。

### 从服务器 DICOM 转换

安装 `requirements-conversion.txt` 中的 Python 依赖，并在服务器准备支持 PNG 的 DCMTK `dcmj2pnm`。转换不需要 PyTorch 或 GPU。

```bash
python -m pip install -r requirements-conversion.txt
dcmj2pnm --version
python main.py convert-dicom \
  --image-metadata-csv /opt/NAS3/datasets/internal/EMBED/tables/metadata_all_cohort_with_ROI_HITI.csv \
  --dicom-column anon_dicom_path \
  --strip-prefix /mnt/NAS2/mammo/anon_dicom \
  --dicom-root /opt/NAS3/datasets/internal/EMBED/anon_dicom \
  --output-dir outputs/dicom_header_check --dry-run --limit 16
```

用户回传的服务器结果已确认，上述前缀替换能找到最先检查的 3 个 2D DICOM，尚未验证全表。保留 `cohort_*/患者/检查/序列/文件.dcm` 后缀，不通过只匹配文件名猜测图像身份。`orig_dicom_path` 指向另一套原始编号，不能用匿名目录直接拼接。

配置已保存为 `configs/embed_conversion_server.json`，可在服务器项目根目录使用 `python main.py convert-dicom --config configs/embed_conversion_server.json` 进行默认 16 张试转换。命令行参数覆盖配置；配置内相对路径以配置目录为基准。仅检查头信息时，追加 `--dry-run --output-dir outputs/dicom_header_check`。旧命令行方式仍然可用。

用户此前还使用过 `/mnt/NAS3/datasets/internal/EMBED/anon_dicom`。是否与 `/opt/NAS3/...` 为链接或同一挂载尚未验证，当前配置使用已实测可读的 `/opt`，不需要先建立这两个根目录的等价关系。

头信息检查通过后，删除 `--dry-run`，并改用独立输出目录（例如 `outputs/mirai_png`）进行 16 张小样本转换。核对图像后，`--limit 0` 表示全量。输出 `images.csv`、`errors.csv`、`conversion.json` 和 PNG；同一输出目录的报告每次重建。已生成 PNG 只有在源文件路径/大小/修改时间、DCMTK 版本及参数一致且图像检查通过时才复用。

转换规则对齐 [Mirai README](https://github.com/yala/Mirai)：`dcmj2pnm +on2 --min-max-window`，输出原尺寸、16 位单通道 PNG，不额外翻转、裁剪、缩放。训练加载器继续执行既有 Mirai 方向及尺寸处理。该固定规则不声称等同于 OncoData 后续版本针对 GE/C-View 的厂商分支。

当前仅选择 `FinalImageType=2D`、L/R、CC/MLO、未标记 spot/mag 的图像；进一步要求 DICOM 为 presentation、单帧、灰度，核对侧别/视图/设备与 CSV 一致。忽略旧 `has_pix_array`、`PNG_flipped` 和 PNG 路径，实际解码失败会记入错误表，不静默改用另一种灰阶算法。DCMTK 负责 DICOM 灰阶显示处理；不要在外部再重复反色。

`images.csv` 包含患者/检查/视图、PNG 绝对路径、原 DICOM 路径和设备型号，可交给 `analysis/mirai_embed/adapt_embed.py export`；该导出器保留设备字段。转换表只是图像候选表，仍需筛查队列、每视图唯一选择、确认结局和患者级分区，不能直接当最终训练 CSV。

本地测试覆盖了合成 DICOM 头解析、路径替换、失败记录、PNG16 校验和设备字段传递；未安装真实 DCMTK，也未访问服务器影像，因此实际灰阶/压缩解码与真实像素仍待服务器小样本核查。

保留原版映射：Selenia Dimensions→0，Lorad Selenia/Hologic Selenia→1，Senograph DS ADS_43.10.1→2，Selenia Dimensions C-View→3。其他型号标为未知，不参与对抗监督，但可参与风险监督。不能将其他 GE 型号随意并入类别 2。

第二阶段要求训练集至少有两个已匹配设备类别；否则明确报错，不静默关闭该分支。未知设备不进入判别器 BatchNorm；判别器更新仅使用已知设备。

### 完整风险因素与缺失监督

可提供 `risk_factors_json`，参考 `data/mirai/risk_factors.example.json`。每次检查一条，factors 使用原版因素名：二值因素是一维 `[0]`/`[1]`；类别因素为原版顺序的 one-hot；缺失使用 `null` 或省略。原版特征名、尺寸会写入 `data_audit.json`。

也可提供 EMBED `clinical_csv`，只映射同一患者、同一检查内一致的 `age_at_study` 与 `tissueden`：年龄按原版 ≤40、≤50、≤60、≤70、≤80、>80 分箱；密度 1–4 映射四分类。冲突或无效值保持缺失，密度 5（男性）不当作女性乳腺密度。显式 risk_factors_json 对应项优先，显式 null 仍保持缺失。

**所有 34 个头都存在。** 缺失项保留固定维度零占位，独立 known mask 防止把占位当阴性标签。第一阶段训练/评估、第二阶段训练按原配置使用给定因素表示；第二阶段评估按原配置使用预测因素。训练 mask_prob=0。

某个因素如果整个训练集都没有标签，其预测头不会得到该因素的辅助监督；保留它不意味着已学会预测它。第二阶段仍保留原版预测因素前向路径，因此无监督头的输出可能影响推理。这是当前数据条件下必须记录的限制，不能将占位和未训练输出解释为真实临床信息。

## 4. 统一操作

```bash
python main.py check-data --config configs/mirai.yaml --pixels --compute-stats
python main.py train --config configs/mirai.yaml --stage all
```

`--stage all` 依次执行：单视图训练 → 第一阶段验证集最佳模型 → 冻结编码器导出 512 维特征 → 四视图与设备对抗训练。阶段失败即停止。不自动对测试集选择模型。

也可分开执行：

```bash
python main.py train --config configs/mirai.yaml --stage 1
python main.py extract-features --config configs/mirai.yaml
python main.py train --config configs/mirai.yaml --stage 2
python main.py evaluate --config configs/mirai.yaml --checkpoint outputs/mirai/stage2/best.pt --split test
python main.py train --config configs/mirai.yaml --stage 2 --resume outputs/mirai/stage2/last.pt
```

第一阶段默认 batch_size=2、accumulate=16，有效批量32；第二阶段默认64、accumulate=1。物理批量按服务器显存调整，保持有效批量。恢复时不允许静默更换批量、学习率、模型配置或数据；可增加 epochs。恢复粒度为**完整 epoch 边界**，不恢复半个 epoch 的未完成工作。开启多工作进程/GPU后仍需实机复核确定性。

输出写入配置的 output_dir：

- `data_audit.json`：排除原因、因素已知数、设备类别、来源指纹。
- `normalization.json`：训练集统计与来源校验。
- `stage1/`、`stage2/`：best.pt、last.pt、history.json、architecture.json。
- `features.pt`：各检查 `[4,512]` 特征与编码器、配置、元数据、影像大小/修改时间指纹。失配则拒绝复用。
- `stage*/test_predictions.csv` 与 `test_metrics.json`：检查级输出与指标。

输出已有 last.pt 时必须显式恢复或换新 output_dir。加载器用于本程序自行生成、可信的 checkpoint，不读取 Mirai 旧完整对象快照。迁移服务器应同步代码、配置、数据、缓存和两个阶段的 checkpoint；跨路径/硬件的恢复在实机验证，不能仅拷贝 stage2 best.pt 后遗漏编码器和特征。

## 5. 与原训练脚本的差异

模型结构保持原版，以下工程或训练差异明确记录：

1. 缺失监督增加 known mask，完整标签时辅助损失与原版一致。
2. 外层训练器明确每次有效主模型更新前执行三次判别器更新；判别器步冻结主模型梯度，主模型步冻结判别器参数及 BatchNorm。原脚本通过 batch step_indx 交替，此处不声称优化轨迹逐步一致。
3. 学习率按验证指标平台下降乘0.1；不实现原脚本降学习率后回滚模型的历史机制，不默认执行全量超参数网格搜索。
4. 训练全量遍历配置队列，不照搬 MGH 每 epoch 的批次数上限。第一阶段按检查平均四张图的概率进行验证。
5. 指标分开输出 `mirai_legacy_c_index` 与 `uno_discrete_c_index`。前者复现源码的“较晚退出时间列 + 原始事件 KM 权重”计算约定并用于选模型；后者使用训练集删失 KM(t-) 和事件年份分数计算离散时间 IPCW concordance。不能把两者混称，也不能当成恢复了精确连续事件时间。无可比较样本/权重支持时为 null，正式训练拒绝静默替代选模指标。年度 AUC 沿用原版纳入规则，单类窗口为 null。

## 6. 小样本验收

```bash
python main.py smoke-test --config configs/mirai.yaml --device cpu --full-resolution
```

生成独立的 `outputs/mirai_smoke/check_*/` 合成数据目录，不覆盖真实输入。验证源码指纹、34头/100维、结构/前向/完整标签损失与梯度、缺失辅助监督、两阶段训练、三比一对抗更新、缓存与原版 MiraiFull.forward 的一致性、保存重载、epoch恢复、患者泄漏与缓存失配拒绝。`--full-resolution` 额外检查原尺寸编码器前向，不代表原尺寸反向或 GPU 显存已通过。

快速合成训练跳过 ImageNet 下载，报告明确标注 synthetic_only；原版 ImageNet 权重下载和映射单独检查。任何合成 AUC/C-index 都没有研究意义。

目前尚未完成真实 DICOM/PNG16 小样本验证、最终 EMBED 队列、GPU 全分辨率反向和大规模训练，不能据工程测试声称论文结果已复现。
