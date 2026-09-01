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

## 五大标准任务模式与置信度等级

| 任务模式 | 可用输入材料 | 锚点定位策略 | 置信度等级 |
| :--- | :--- | :--- | :--- |
| **Task A: `NATIVE_ROI_MATCH`** | 仅需 WSI (提供坐标/选框) | 精确 Level-0 物理坐标映射（**0 锚点反查误差**） | **Tier A (极高/精确)** |
| **Task B: `SINGLE_CROP_REPRODUCE`** | WSI + 4× 历史截图 | SIFT 多角度旋转搜索 + 物理尺度 NCC 模板回退 | **Tier C (中等)** |
| **Task C: `DUAL_SCALE_REPRODUCE`** | WSI + 4× 与 20× 截图 | 4× 粗候选检索 + 20× 独立高倍几何核验 | **Tier B (高/严格复现)** |
| **Task D: `HIGH_MAG_ASSISTED`** | WSI + 仅有 20× 截图 | 高倍视场辅助引导搜索 | **Tier D (需人工指引)** |
| **Task E: `WHOLE_SLIDE_REGISTER`** | 多染色 WSI (无任何截图) | 全片间建立几何变换图，输出对齐概览 | **Tier A (全景精确)** |

---

## 命令行使用指南 (CLI)

```bash
# 1. 资产自动发现 (Discover)
crossstainwsi discover --base-dir /path/to/wsi --tiff-dir /path/to/crops

# 2. 执行计划预览 (Plan)
crossstainwsi plan 4W-5-3 --base-dir /path/to/wsi --tiff-dir /path/to/crops

# 3. 单样本执行 (Run)
# 最简运行 (全默认参数)
crossstainwsi run 4W-5-3

# 指定基准染色、目标染色与输出 DPI (例如 600 DPI)
crossstainwsi run 4W-5-3 --ref-stain HE --stains HE Gram --dpi 600

# 水平镜像截图纠偏
crossstainwsi run 2-2W-1 --mirror

# 4. 批量执行 (Batch)
crossstainwsi batch 4W-5-3 2-2W-1 3-4W-2 --out-dir /path/to/output
```

### CLI 核心参数清单

| 参数 | 默认值 | 作用与说明 |
| :--- | :--- | :--- |
| `--ref-stain` | `masson` | 指定参考基准染色（截图是从哪张切片上截取的） |
| `--stains` | `HE Gram` | 指定需要对齐提取的移动染色列表 |
| `--dpi` | `300` | 输出出版级 TIFF 的分辨率 (DPI) |
| `--scale-ratio` | `5.0` | 20× 相对 4× 的视场采样倍率比例 |
| `--mirror` | `False` | 强制对输入的截图进行水平翻转纠偏 |
| `--no-overlay` | `False` | 关闭半透明配准重叠图 (Overlay) 输出 |
| `--contact-sheet`| `False` | 开启多染色横向对比拼图输出 (默认关闭) |

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
        └── registration_report.json
```

---

## 单元测试

```bash
pytest -v
```

---

## 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
