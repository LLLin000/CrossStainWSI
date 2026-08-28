"""
产物安全隔离与质量审核状态体系 (Review & Safety States)
"""

from enum import Enum
from pathlib import Path
from typing import Tuple


class ArtifactTier(str, Enum):
    """
    产物输出层级：
    - FINAL: 满足所有黄金标准，可直接用于论文插图和定量分析
    - REVIEW: 处于临界置信区或存在轻微序列差异，需要人工在审核界面确认
    - DEBUG: 未通过质控、ABSTAIN 或失败，仅保留诊断图像与日志，绝不生成虚假出版图
    """
    FINAL = "final"
    REVIEW = "review"
    DEBUG = "debug"


class ConfidenceTier(str, Enum):
    """
    输入证据与锚点置信度等级：
    - TIER_A_NATIVE: 用户直接在 Reference WSI 上框选 Level-0 ROI (锚点置信度 100%, 无反查误差)
    - TIER_B_DUAL_SCALE: 拥有 4x + 20x 双尺度截图并经过 4x 定位 + 20x 独立验证
    - TIER_C_SINGLE_CROP: 仅有 4x 截图 (大体定位可靠，但无高倍独立验证)
    - TIER_D_HIGH_MAG_ASSISTED: 仅有 20x 截图 (缺乏全局上下文，需人工指引或辅助)
    - TIER_E_AMBIGUOUS: 存在多处对称候选或无法唯一锁定，强制 ABSTAIN
    """
    TIER_A_NATIVE = "TIER_A_NATIVE"
    TIER_B_DUAL_SCALE = "TIER_B_DUAL_SCALE"
    TIER_C_SINGLE_CROP = "TIER_C_SINGLE_CROP"
    TIER_D_HIGH_MAG_ASSISTED = "TIER_D_HIGH_MAG_ASSISTED"
    TIER_E_AMBIGUOUS = "TIER_E_AMBIGUOUS"


class RunVerdict(str, Enum):
    """
    全流程执行最终裁决
    """
    PASS = "PASS"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"
    INCOMPLETE = "INCOMPLETE"  # 缺少必选染色切片
    FAIL = "FAIL"


def resolve_artifact_dir(sample_out_dir: Path, verdict: RunVerdict) -> Path:
    """
    根据裁决结果将输出文件路由到对应的子目录，防止将低质量/失败图混入正式出版目录
    """
    sample_out_dir = Path(sample_out_dir)
    if verdict == RunVerdict.PASS:
        target = sample_out_dir / ArtifactTier.FINAL.value
    elif verdict == RunVerdict.REVIEW:
        target = sample_out_dir / ArtifactTier.REVIEW.value
    else:
        target = sample_out_dir / ArtifactTier.DEBUG.value

    target.mkdir(parents=True, exist_ok=True)
    return target
