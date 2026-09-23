# CortexAdapt3D

基于皮层表面几何与顶点属性的 dHCP 早产分类代码。每个左右半球作为一个样本，使用 `sulc`、`curv` 和 `thickness` 三个顶点属性；标签 `1` 表示早产，`0` 表示足月。本仓库整理自服务器上的 `Prompt_Tuning/Tuning/cls` 分类实验代码。

## 来源与发布范围

本项目在 [PointGST](https://github.com/jerryfeng2003/PointGST) 基础上修改，保留其 Apache 2.0 许可证及 [署名说明](NOTICE.md)。当前发布范围是 dHCP 分类任务。原目录中的 ModelNet/ScanObjectNN 基准、补全与检测任务、未接入分类入口的回归配置和实验产物没有纳入。仓库不包含被试数据、划分文件、预训练权重或训练检查点。

## 环境

建议使用 Python 3.9+ 和可用的 PyTorch CUDA 环境：

```bash
pip install -r requirements.txt
```

PyTorch 与 CUDA 的组合应按运行机器选择。本项目不再依赖原 PointGST 仓库中用于通用点云基准的 PointNet++、KNN CUDA、Chamfer Distance 或 EMD 扩展。

## 数据

`--data-root` 下应有 `sub-<participant>/ses-<session>/anat/sub-<participant>_hemi-L_surface_ico5.vtk` 及对应右半球文件。每个表面应有 10,242 个顶点和 `sulc`、`curv`、`thickness` 顶点属性。

`--csv-path` 是制表符分隔的被试表，至少包含 `participant_id`、`session_id`、`birth_age`、`scan_age` 列。`--split-root` 下应有 `dhcp_train_ids_seed42.txt`、`dhcp_val_ids_seed42.txt` 和 `dhcp_test_ids_seed42.txt`，每行一个样本 ID，格式为 `sub-<participant>_ses-<session>_left` 或 `_right`。三个划分不能重叠。

## 训练与测试

从仓库根目录运行。训练需要兼容的 Point-MAE 预训练检查点，索引文件已随代码提供。

```bash
python main.py \
  --data-root /path/to/dhcp \
  --csv-path /path/to/combined_03.csv \
  --split-root /path/to/splits \
  --ckpts /path/to/pretrained.pth \
  --exp-name dhcp_preterm
```

最佳验证准确率对应的权重保存于 `experiments/finetune_dHCP_classification/dhcp_preterm/ckpt-best.pth`。仅在训练结束后使用该权重评估测试集，并导出 `metrics.csv` 和 `predictions.csv`。

```bash
python main.py \
  --data-root /path/to/dhcp \
  --csv-path /path/to/combined_03.csv \
  --split-root /path/to/splits \
  --ckpts experiments/finetune_dHCP_classification/dhcp_preterm/ckpt-best.pth \
  --test --exp-name dhcp_test
```

测试检查点必须包含本项目训练时保存的三个属性的均值和标准差。原服务器中的旧检查点可能不包含这些统计量，需要重新训练或单独转换后才能使用新的测试入口。

## 说明

模型注册名为 `CortexAdapt3D`，谱域模块为 `SpectralAdapter`。为兼容已有预训练参数，部分内部参数键仍保留上游命名。这里不公布新的性能数字；正式结果应在相同数据划分和检查点下重新测定。
