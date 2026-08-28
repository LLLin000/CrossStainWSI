# CrossStainWSI (多染色病理切片自适应配准与出版级提取引擎)

[English](README.md) | [简体中文](README_zh.md)

**CrossStainWSI** 是一个面向数字病理学与生物医学研究的**可审计、自适应跨染色全切片图像 (WSI) 自动配准与多尺度同区域提取工具包**。

致力于解决连续组织切片因不同染色（如 Masson、HE、Gram、IHC 免疫组化等）在切片旋转、位移、轻微形态差异及染色灰度巨大反差下，难以精确锁定并提取同一微观解剖视野的核心痛点。

---

## 核心特性

- **输入证据与输出要求彻底解耦**：
  - **输入材料（可选）**：支持仅有 WSI 切片、仅有 4× 截图、4×+20× 双尺度截图、或原生 WSI Level-0 框选坐标。绝不再强制要求固定截图输入。
  - **输出要求（按需）**：支持按需提取 4×、20×、10×、40× 或任意指定物理微米视场（$\mu\text{m} \times \mu\text{m}$）的高保真图像。
- **自适应工作流规划器 (Workflow Planner)**：
  - 自动探测已有数据材料与用户目标，智能撮合并生成明确的 `ExecutionPlan`，运行前即可直观预览任务类别与置信度等级。
- **纯解剖形态学深度配准**：
  - 结合连通组织岛隔离匹配、多角度 LoFTR 深度形态特征对齐与局部微残差自适应微调（Local LoFTR + Sobel 梯度相位相关）。
- **单次 Level-0 逆映射无畸变重采样**：
  - 严格通过数学复合矩阵直接从原始切片最高分辨率层（Level 0）执行单次逆映射采样（`WARP_INVERSE_MAP`），彻底杜绝多次旋转插值带来的画质模糊、非等比拉伸、剪切形变与人工白边。
- **严格产物安全隔离与质控闭环 (Safety Gating)**：
  - 产物自动分流至 `final/`（全项达标）、`review/`（临界质量需人工复核）与 `debug/`（放弃/失败）。**当证据不足或关键染色缺失时，严禁输出虚假出版图**。

---

## 系统架构

```text
               用户 / 数据资产目录
                        │
                        ▼
             ┌─────────────────────┐
             │   Asset Discovery   │  (资产发现: 自动扫描 WSI、截图证据、坐标标注)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Workflow Planner   │  (工作流规划: 撮合 UserGoal 与物理采集先验)
             └──────────┬──────────┘
                        │
                        ▼
                 ExecutionPlan     (生成自包含执行计划: TaskType, ConfidenceTier)
                        │
                        ▼
             ┌─────────────────────┐
             │  Registration Core  │  (配准核心: 组织岛隔离 + 全局LoFTR + 局部微调)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Transform Graph   │  (坐标拓扑图: 统一解析复合矩阵 M_20x_to_L0)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Level-0 Sampling   │  (WSISampler 单次全精度逆映射重采样)
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Artifact Routing   │  (产物安全分流: final/ vs review/ vs debug/)
             └─────────────────────┘
```

---

## 五大标准任务模式与置信度等级

| 任务模式 | 可用输入材料 | 锚点定位策略 | 置信度等级 |
| :--- | :--- | :--- | :--- |
| **Task A: `NATIVE_ROI_MATCH`** | 仅需 WSI (提供坐标/选框) | 精确 Level-0 物理坐标映射（**0 锚点反查误差**） | **Tier A (极高/精确)** |
| **Task B: `SINGLE_CROP_REPRODUCE`** | WSI + 4× 历史截图 | SIFT 多角度旋转搜索 + 物理尺度 NCC 模板回退 | **Tier C (中等)** |
| **Task C: `DUAL_SCALE_REPRODUCE`** | WSI + 4× 与 20× 截图 | 4× 粗候选检索 + 20× 独立高倍几何核验 | **Tier B (高/严格复现)** |
| **Task D: `HIGH_MAG_ASSISTED`** | WSI + 仅有 20× 截图 | 高倍视场辅助引导搜索 | **Tier D (需人工指引)** |
| **Task E: `WHOLE_SLIDE_REGISTER`** | 多染色 WSI (无任何截图) | 全片间建立几何变换图，输出对齐概览 | **Tier A (全景精确)** |

---

## 安装说明

```bash
# 克隆仓库
git clone https://github.com/LLLin000/CrossStainWSI.git
cd CrossStainWSI

# 可编辑模式安装
pip install -e .
```

### 核心环境依赖
- Python >= 3.10
- PyTorch & Kornia (用于 LoFTR 深度形态学匹配模型)
- OpenCV (`cv2`)
- `kfbslide` 或 `OpenSlide` (用于 WSI 切片高速 I/O)
- Pillow, NumPy, SciPy

---

## 命令行使用指南 (CLI)

### 1. 资产自动发现 (Discover)
自动扫描指定目录下的切片与截图证据，输出清晰概览：
```bash
crossstainwsi discover --base-dir /path/to/wsi --tiff-dir /path/to/crops
```

### 2. 执行计划预览 (Plan)
在运行计算密集型配准之前，直观查看引擎生成的执行计划：
```bash
crossstainwsi plan 4W-5-3 --base-dir /path/to/wsi --tiff-dir /path/to/crops
```

### 3. 单样本执行 (Run)
执行配准、采样并自动根据质控等级将结果归档至 `final/`、`review/` 或 `debug/`：
```bash
crossstainwsi run 4W-5-3 --base-dir /path/to/wsi --out-dir /path/to/output
```

### 4. 批量执行 (Batch)
安全批量处理多个样本（支持断点续跑，每次完成实时更新 `batch_summary.json`，**绝不自动关机**）：
```bash
crossstainwsi batch 4W-5-3 2-2W-1 3-4W-2 --out-dir /path/to/output
```

---

## Python API 调用示例

```python
from pathlib import Path
from crossstainwsi.pipeline import PipelineConfig, SampleRunner, BatchRunner
from crossstainwsi.planning import UserGoal, ViewSpec, StainRequirement

# 1. 自定义业务目标
goal = UserGoal(
    reference_stain="masson",
    stain_requirements=[
        StainRequirement("HE", is_required=True),     # HE 为必选（缺失则中断）
        StainRequirement("Gram", is_required=False),  # Gram 为可选
    ],
    requested_views=[
        ViewSpec(name="4x", pixel_dimensions=(2257, 1310), magnification_approx=4.0),
        ViewSpec(name="20x", pixel_dimensions=(2257, 1310), magnification_approx=20.0),
    ],
)

# 2. 执行单样本
cfg = PipelineConfig(
    base_dir=Path(r"E:\研究数据\骨科\切片扫描\2026-08-21"),
    output_dir=Path(r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi"),
)
runner = SampleRunner(config=cfg, goal=goal)
report = runner.process("4W-5-3")

print(f"Overall Status: {report['overall_status']}")
print(f"Artifact Tier:  {report['artifact_tier']}")
```

---

## 产物目录规范

输出文件严格按审核状态分流存放：

```text
output_dir/
└── 4W-5-3/
    └── final/                   # 仅在全部 QC 通过时生成
        ├── 4W-5-3-Masson-4x-300dpi.tif
        ├── 4W-5-3-Masson-20x-300dpi.tif
        ├── 4W-5-3-HE-4x-aligned-300dpi.tif
        ├── 4W-5-3-HE-20x-aligned-300dpi.tif
        ├── overlay-HE-4x-aligned.png
        ├── overlay-HE-20x-aligned.png
        ├── contact_sheet_4x.png
        ├── contact_sheet_20x.png
        └── registration_report.json
```

---

## 单元测试

运行项目完整测试套件：
```bash
pytest -v
```

---

## 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
