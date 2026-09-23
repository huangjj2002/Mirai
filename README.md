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

## 上游来源

模型代码来自 [yala/Mirai](https://github.com/yala/Mirai)，固定提交 `3904a9eaca046378a194b1eb8c62fa32f45ce83b`。所需源码、原始 MIT 许可证和校验清单保留在 `mirai_training/vendor/`。上游代码许可不表示本仓库包含 EMBED 数据或提供其使用授权。
