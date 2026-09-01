# CrossStainWSI — 产品需求与技术规格文档 (PRD & Technical Specification)

> **版本**：v0.3.0-draft  
> **状态**：Approved for Implementation  
> **核心定位**：可审计、自适应、多模态全切片图像 (WSI) 自动配准与出版级同区域提取引擎

---

## 1. 执行摘要与产品愿景 (Executive Summary)

### 1.1 核心痛点
在生物医学研究（特别是骨科软骨/骨缺损、肿瘤微环境、空间生物学）中，连续组织切片因经历不同成像与染色协议（明场 Masson/HE/Gram/番红固绿、IHC 免疫组化 DAB、多色免疫荧光 mIF/CyCIF、空间转录组芯片）：
1. 存在由于切片旋转、位移、轻微形态变异与灰度/光谱反差导致的**同解剖微观视野难以对齐提取**问题；
2. 现有开源学术工具（如 VALIS, PALOM）缺乏面向生物/医学科研人员的零门槛交互体系，且非刚性弹性形变（B-spline）容易人为改变骨小梁物理厚度与软骨孔隙结构；
3. 传统图像处理工具（ImageJ, TrakEM2）缺乏全切片金字塔支持，动辄内存溢出，人工打点耗时耗力。

### 1.2 产品使命
构建一个**轻量、确定性高、刚性解剖保真、输入证据与输出要求解耦、并具备严格质量分级（final/review/debug）**的跨模态切片配准系统。

---

## 2. 核心架构与设计原则 (Core Principles)

```text
               任意切片输入 (WSI / 截图证据 / 原生坐标)
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │   Modality Adapter   │
                     └──────────┬───────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │   Canonical Structural Representation Set      │
        │  (Mask, Contour, Gradient, Nuclear Field, ...) │
        └───────────────────────┬────────────────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Informativeness    │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   Candidate Search   │  (Multi-angle, Scale, Island)
                     └──────────┬───────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │        Pluggable Multi-Matcher Backends        │
        │   [Feature: LoFTR] [Structure: Phase/Edge]     │
        │   [Information-Theoretic: NMI Basin Rescue]    │
        └───────────────────────┬────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │     Evidence Fusion & Strict Safety Gating     │
        │             (PASS / REVIEW / ABSTAIN)          │
        └───────────────────────┬────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │   Anatomical Retrieval (Rigid / Similarity)    │
        │     Single-Pass Level-0 Inverse Warp Sampling  │
        └───────────────────────┬────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │     Artifact Routing & Publication Output      │
        │       (final/ vs review/ vs debug/ 300 DPI)    │
        └────────────────────────────────────────────────┘
```

### 核心不变量 (Invariants)
1. **解耦性**：输入证据（WSI、4×/20×截图、原生坐标）与输出需求（4×、10×、20×、40× 任意物理微米视场）彻底解耦；
2. **多表征集而非单一灰度图**：Adapter 输出结构表征集（Representation Set），由匹配器根据两边共同存在的信息决定匹配通路；
3. **刚性解剖保真度**：用于最终出版图提取的几何变换严格限制为：
   $$\text{Reflection} + \text{Rotation} + \text{Translation} + \text{Isotropic Scale Prior}$$
   严禁使用非刚性形变改变骨小梁或组织微观厚度；
4. **单次 Level-0 逆重采样**：最终切片提取直接由 `WSISampler` 通过复合矩阵一次性从原始金字塔最高分辨率层采样，严禁多次插值模糊与人工白边；
5. **质量安全门禁**：质控不合格或缺少必选染色时判定为 `ABSTAIN` / `INCOMPLETE`，绝不在 `final/` 目录生成虚假出版图。

---

## 3. 数据模型与表征规范 (Data Contracts)

### 3.1 规范结构表征集 (`CanonicalRepresentation`)

```python
@dataclass
class CanonicalRepresentation:
    # 1. 空间有效性与掩模
    valid_mask: np.ndarray                 # 有效成像区域布尔掩模 (非背景/非暗场死区)
    tissue_mask: Optional[np.ndarray]      # 组织实质前景掩模
    artifact_mask: Optional[np.ndarray]    # 气泡、折皱、笔迹、强饱和光斑等瑕疵掩模

    # 2. 宏观几何与轮廓
    coarse_contour: Optional[np.ndarray]   # 宏观组织外轮廓多边形/二值边
    distance_field: Optional[np.ndarray]   # 组织外边缘欧氏距离变换场 (EDT)

    # 3. 结构梯度与特征
    gradient_pyramid: Tuple[np.ndarray, ...] # 多尺度 Sobel/DoG 结构梯度幅值
    nuclear_density: Optional[np.ndarray]  # 跨模态同源核结构场 (Nuclear Density/Response)
    feature_image: Optional[np.ndarray]    # 供 LoFTR 等深度特征网络直接提取的图

    # 4. 物理元数据与溯源
    mpp_xy: Optional[Tuple[float, float]]  # 物理分辨率 (微米/像素)
    modality: str                          # "brightfield", "ihc_dab", "fluorescence", "generic"
    representation_provenance: Dict[str, Any] # 记录核结构场来源 (Hematoxylin / DAPI / LoG)
    informativeness: Dict[str, float]      # 图像信息量/信噪比指标
```

### 3.2 模态适配器规范 (Modality Adapters)

#### ① 明场组织学适配器 (`GenericBrightfieldAdapter`)
- **适用**：Masson、HE、Gram、番红固绿、天狼星红、甲苯胺蓝、PAS、Goldner 等；
- **处理流水线**：
  $$\text{RGB} \to \text{Background Subtraction} \to \text{CLAHE Normalization} \to \text{Gradient Pyramid \& Contour}$$

#### ② 免疫组化适配器 (`IHCDeconvolutionAdapter`)
- **适用**：H&E ↔ IHC-DAB (如 CD68, Col II, Ki67, ER, PR, HER2)；
- **核心算法**：Ruifrok-Johnston 色彩解卷积（Beer-Lambert 光学密度分离）：
  - 提取 **Hematoxylin (苏木精)** 光学密度场作为 `nuclear_density`；
  - 提取 **DAB** 棕色通道生成掩模（在配准中降权，显著降低 DAB 强阳性对几何结构的干扰）；
  - 配准退化为：$$\text{Hematoxylin Nuclear Field}_{\text{HE}} \longleftrightarrow \text{Hematoxylin Nuclear Field}_{\text{IHC}}$$

#### ③ 荧光/暗场适配器 (`FluorescenceAdapter`)
- **适用**：暗场多色免疫荧光 (mIF/CyCIF)、明场 H&E ↔ 荧光切片；
- **核心算法**：
  - 自动识别并提取 **DAPI 细胞核通道**；
  - 通过背景扣除 + **LoG (Laplacian of Gaussian)** / **DoG** 滤波器计算细胞核中心概率响应；
  - 配准退化为跨模态同源核结构场匹配：$$\text{Hematoxylin Response}_{\text{Brightfield}} \longleftrightarrow \text{DAPI Nuclear Response}_{\text{Fluorescence}}$$

---

## 4. 三层匹配器架构 (Three-Tier Matcher Backends)

引擎将匹配能力解耦为三级可插拔后端，严禁将单一 LoFTR 视为万能解：

```text
┌─────────────────────────────────────────────────────────────┐
│ A. Feature Matcher (特征匹配通路)                            │
│    - LoFTR 密集 Transformer 特征匹配                        │
│    - SIFT 多角度旋转特征提取                                │
│    - 作用：明场同模态、低跨模态差异切片的快速全局检索        │
├─────────────────────────────────────────────────────────────┤
│ B. Structure Matcher (结构匹配通路)                          │
│    - 组织外轮廓与欧氏距离场 (Distance Field) 匹配           │
│    - Sobel 梯度幅值相位相关 (Phase Correlation)             │
│    - 作用：微小位移残差锁定、宏观形状对齐                   │
├─────────────────────────────────────────────────────────────┤
│ C. Information-Theoretic Matcher (信息论局部优化救援通路)   │
│    - 归一化互信息 (Normalized Mutual Information, NMI)       │
│    - 多变量互信息 (Multivariate MI)                         │
│    - 限制：必须依赖前级候选区域 (Basin of Attraction) 启动  │
│    - 作用：跨模态灰度剧烈反差下的高精度局部微调             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 失败分类体系与质控评估 (Failure Taxonomy & QC)

系统摒弃单一的“成功/失败”二元标签，采用细粒度的失败诊断树：

### 5.1 失败分类代码表 (Failure Taxonomy)
- `REFERENCE_ANCHOR_FAIL`：参考切片手工截图无法在 WSI 中唯一定位；
- `MIRROR_AMBIGUOUS`：无法确定输入是否发生水平镜像翻转；
- `ROTATION_AMBIGUOUS`：旋转角度存在对称多峰二义性；
- `LOW_INFORMATION`：切片区域组织覆盖极低（纯空白玻片或无细胞背景）；
- `LOW_OVERLAP`：两张切片重叠面积不足，无法建立有效几何关联；
- `FEATURE_MATCH_WEAK`：LoFTR / SIFT 内点数不足；
- `STRUCTURE_CONFLICT`：轮廓距离场与特征点对齐方向相互矛盾；
- `CROSS_SCALE_CONFLICT`：4× 预测位置与 20× 独立核验位置不一致；
- `MODALITY_ADAPTER_FAIL`：色彩解卷积或 DAPI 通道提取异常；
- `LOCAL_REFINEMENT_FAIL`：局部残差微调发散；
- `BIOLOGICAL_SECTION_MISMATCH`：连续切片物理间隔过大导致的生物学解剖漂移。

### 5.2 质量等级与产物分流规则
- **`PASS` $\to$ 路由至 `final/`**：
  - 必须满足：全部必选染色存在 + 锚点可信 + 全局尺度 $0.95 \le s \le 1.05$ + 内点数 $\ge 35$ (内点率 $\ge 12\%$) + 空间覆盖率 $\ge 20\%$ + 20× 核验一致。
- **`WARN` / `REVIEW` $\to$ 路由至 `review/`**：
  - 处于临界置信区，或因跳层存在轻微切面差异，需人工复核。
- **`ABSTAIN` / `INCOMPLETE` $\to$ 路由至 `debug/`**：
  - 触发上述失败分类，**严禁在 `final/` 生成任何虚假出版 TIFF**，仅保留诊断日志。

---

## 6. 测试与 Benchmark 验证框架 (Evaluation Strategy)

评测体系分为四层梯队，避免用肉眼主观判断代替严格度量：

### 6.1 四级评测数据集体系
1. **Tier 1: Synthetic Benchmark (合成真值基准)**
   - 对真实切片施加精确已知的几何变换：$$\theta \in [-180^\circ, 180^\circ],\quad (dx, dy),\quad s \in [0.9, 1.1],\quad \text{mirror} \in \{\text{True}, \text{False}\}$$
   - 验证目标：绝对几何误差 $E_\theta < 0.5^\circ$, $E_{xy} < 2\text{ px}$, $\text{Mirror Accuracy} = 100\%$。
2. **Tier 2: Multi-Modal Pathology Slices (多模态实测数据)**
   - 使用已下载的 `benchmarks/ihc/` (OME-TIFF 连续切片) 与 `benchmarks/cycif/` (多通道荧光) 进行适配器回归测试。
3. **Tier 3: Gigapixel WSI Full-Slide (全切片金字塔基准)**
   - 本地 30 组骨科 KFB 全幅切片 ($60000 \times 48000\text{ px}$)，验证全图组织岛隔离、坐标图复合求解与 Level-0 逆重采样吞吐性能。
4. **Tier 4: Public Landmark Challenges (国际金标准挑战赛)**
   - 接入 ACROBAT (37,000+ 标注点) 与 ANHIR 数据集，评估真实目标配准误差（Target Registration Error, TRE）：
     $$rTRE = \frac{\|T(p_i) - q_i\|}{\text{Image Diagonal}}$$
     报告 Median TRE, IQR 与 Success Rate @ $50\,\mu\text{m}$。

---

## 7. 实施路线图与里程碑 (Milestones M0 ~ M7)

```text
┌─────────────────────────────────────────────────────────────┐
│ M0: Benchmark Harness & Synthetic Ground Truth              │
│     - 合成几何扰动测试集生成器                               │
│     - Failure Taxonomy 诊断树与误差统计器                   │
├─────────────────────────────────────────────────────────────┤
│ M1: CanonicalRepresentation Contract & Brightfield          │
│     - 冻结 RepresentationSet 数据结构与掩模生成器            │
│     - 通用明场适配器 (GenericBrightfieldAdapter)             │
├─────────────────────────────────────────────────────────────┤
│ M2: IHC Deconvolution Adapter                               │
│     - 基于 Ruifrok-Johnston 的 H/DAB 通道解卷积             │
│     - H ↔ H 同源核结构场提取                                 │
├─────────────────────────────────────────────────────────────┤
│ M3: Fluorescence & Multi-Channel Adapter                    │
│     - DAPI 自动提取与 LoG 细胞核响应场计算                   │
│     - 暗场 ↔ 暗场 及 明场 ↔ 荧光同源配准                    │
├─────────────────────────────────────────────────────────────┤
│ M4: Pluggable Matcher Backends                              │
│     - LoFTR Feature Matcher                                 │
│     - Gradient Phase & Distance Field Matcher               │
│     - Normalized Mutual Information (NMI) Basin Optimizer   │
├─────────────────────────────────────────────────────────────┤
│ M5: Evidence Fusion & Quality Gating                        │
│     - 多假设候选重排与打分                                   │
│     - final / review / debug 产物严格路由                   │
├─────────────────────────────────────────────────────────────┤
│ M6: Real Benchmark Validation                               │
│     - 在 benchmarks/ihc 和 benchmarks/cycif 上实测           │
│     - 绘制消融实验对比表 (Ablation Table)                    │
├─────────────────────────────────────────────────────────────┤
│ M7: Gigapixel Full Pipeline & Standalone Delivery           │
│     - 30 组全切片端到端回归                                  │
│     - PyInstaller 单文件独立运行打包                         │
└─────────────────────────────────────────────────────────────┘
```
