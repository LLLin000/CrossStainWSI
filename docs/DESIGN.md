# CrossStainWSI — 系统架构与详细设计规范 (Design Specification)

## 1. 产品使命与核心设计原则

CrossStainWSI 是一个面向数字病理学与生物医学研究的可审计、自适应跨染色全切片图像 (WSI) 自动配准与多尺度同区域提取工具包。

### 核心设计原则

1. **输入证据与输出要求彻底解耦**
   - **输入证据（可选）**：包括各染色 WSI 切片、已有截图（任意放大倍率）、或原生 Level-0 框选坐标。绝不再强制要求固定必须提供 4× 和 20× 截图。
   - **输出要求（按需）**：用户直接按需勾选需要导出的倍率视野（`4x`, `10x`, `20x`, `40x` 或指定物理微米视场），系统自动直接从原始最高清层（Level 0）单次采样生成。

2. **约定优于配置 (Convention over Configuration)**
   - 采用标准、严谨的文件命名约定；
   - 不做复杂的低效手动映射界面，避免人工拉下拉框配错切片导致科研结论事故；
   - 提供智能“文件体检诊断”（`crossstainwsi discover`），对未规范命名的切片给出清晰的修改提示。

3. **单次 Level-0 逆映射无畸变重采样**
   - 计算在降采样层完成，但所有最终切片提取严格使用 OpenCV 逆映射矩阵 (`WARP_INVERSE_MAP`) 一次性直接从 WSI Level 0 原始像素重采样；
   - 严禁对有限中间图像进行多次旋转与二次裁剪，彻底杜绝插值模糊、非等比拉伸、剪切形变与人工黑白边。

4. **严格产物安全隔离 (Safety Gating)**
   - 只有满足全部黄金质控标准的切片才进入 `final/`；
   - 临界质量或轻微切面差异进入 `review/`；
   - 质控不合格、存在二义性或缺少必选染色时判定为 `ABSTAIN`/`INCOMPLETE` 并归档于 `debug/`，**绝不生成虚假的最终出版 TIFF 图**。

---

## 2. 文件命名规范与体检诊断

### 2.1 唯一定名标准

```text
切片 WSI 文件： {样本名}-{染色名}.kfb  (或 .svs / .ndpi / .mrxs)
用户截图文件：   {样本名}-{倍率}.tif     (或 {样本名}-{染色名}-{倍率}.tif)
```

#### 命名示例：
- `3-4w-2-masson.kfb`
- `3-4w-2-HE.kfb`
- `3-4w-2-Gram.kfb`
- `3-4w-2-4x.tif`（或 `3-4w-2-masson-4x.tif`）
- `3-4w-2-20x.tif`（或 `3-4w-2-masson-20x.tif`）

> **自动推导特性**：如果截图文件名中包含了染色名（如 `3-4w-2-masson-4x.tif`），引擎会自动将 `masson` 识别为基准参考染色，用户无需手动指定 `--ref-stain`。

### 2.2 格式兼容性
- **WSI 格式**：支持 `.kfb` (kfbslide), `.svs` (Aperio), `.ndpi` (Hamamatsu), `.mrxs` (3DHISTECH), `.tif` / `.tiff` (BigTIFF)。
- **截图格式**：支持 `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` 任意格式。

---

## 3. 通用多尺度“双保险”验证机制

系统支持任意倍率的截图输入（例如 2×, 4×, 10×, 20×, 40×）：

1. **单张截图输入**：
   - 执行全局定位（SIFT 多角度搜索 + 物理尺度 NCC 模板匹配回退）；
   - 置信度等级为 `Tier C (Single Crop)`。
2. **两张不同倍率截图输入（双保险闭环）**：
   - 引擎自动比对视场大小，识别出较宽的低倍图与局部的较高倍图；
   - **低倍图（宽视场）**：负责大范围粗测定位、确定属于哪块组织岛、判断旋转角度与消除全局搜索二义性；
   - **高倍图（微观视场）**：在低倍预测的局部区域执行独立几何核验与微调，确保骨小梁/微观细胞结构完全咬合，排除对称伪峰；
   - 置信度等级提升为 `Tier B (Dual Scale)`。
3. **原生 WSI 坐标输入**：
   - 用户直接输入中心点坐标 $(x, y)$ 或在看片软件中选定坐标；
   - **0 锚点反查误差**，置信度等级为 `Tier A (Native Exact)`。

---

## 4. 输出倍率选择与坐标拓扑图 (Transform Graph)

用户无需关心内部采样比计算，在 CLI 或 GUI 中直接指定目标倍率（如 `-m 4x 10x 20x`）：

### 4.1 几何复合矩阵推导

坐标图统一维护以下 3×3 齐次变换矩阵：

$$M_{view \to \text{moving\_L0}} = S_{\text{L2} \to \text{L0}} \cdot M_{\text{crop4} \to \text{moving\_L2}} \cdot M_{\text{local}}^{-1} \cdot M_{view \to \text{crop4}}$$

其中：
- $M_{view \to \text{crop4}}$ 由目标视图与主参考视图的放大倍率比例和对齐先验自动生成；
- $M_{\text{local}}^{-1}$ 补偿局部 LoFTR / 相位相关微调残差；
- $M_{\text{crop4} \to \text{moving\_L2}}$ 承载跨染色全局多角度匹配矩阵与组织岛偏移；
- $S_{\text{L2} \to \text{L0}}$ 直接将尺度提升至最高清层 Level 0。

---

## 5. 产物安全分流与质量控制体系

输出目录结构严格按审核状态分流：

```text
output_dir/
└── 4W-5-3/
    ├── final/                   # 仅当全部 QC 通过时生成 (PASS)
    │   ├── 4W-5-3-Masson-4x-300dpi.tif
    │   ├── 4W-5-3-Masson-20x-300dpi.tif
    │   ├── 4W-5-3-HE-4x-aligned-300dpi.tif
    │   ├── 4W-5-3-HE-20x-aligned-300dpi.tif
    │   ├── overlay-HE-4x-aligned.png
    │   └── registration_report.json
    │
    ├── review/                  # 临界质量或轻微切面差异需人工复核时存放 (WARN)
    │   └── ...
    │
    └── debug/                   # 质控失败、ABSTAIN 或缺少必选染色时存放 (ABSTAIN / INCOMPLETE)
        └── registration_report.json  (仅记录诊断日志，不生成虚假出版 TIFF)
```

### 5.1 质控黄金标准
- **内点数 (Inliers)** $\ge 35$ 且内点率 (Inlier Ratio) $\ge 12\%$
- **空间覆盖率 (Spatial Coverage)** $\ge 20\%$ (基于 4×4 网格占用率)
- **等比缩放约束 (Scale)** $0.95 \le \text{scale} \le 1.05$ (禁止非等比拉伸与剪切形变)
- **重投影中位数误差 (Median Reprojection Error)** $< 5.0\text{ px}$

---

## 6. CLI 与未来 GUI 架构契约

### 6.1 CLI 命令集
- `crossstainwsi discover`: 资产自动探测与输入体检；
- `crossstainwsi plan <sample>`: 执行计划预览（免跑重型计算，直接查看调度策略）；
- `crossstainwsi run <sample>`: 单样本执行配准与分级提取；
- `crossstainwsi batch <samples...>`: 批量运行（断点安全落盘，无自动关机）。

### 6.2 GUI 对接契约
底层 `PipelineConfig` 提供 `progress_callback(sample_id, pct, msg)` 标准回调，数据模型 `AssetInventory` 和 `ExecutionPlan` 能够直接绑定到图形界面表格与进度条，无需任何算法重构即可快速封装单文件独立运行 `.exe`。
