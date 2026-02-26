# models/fusion/__init__.py

from .fusion_module import (
    FusionConfig,
    UncertaintyGuidedFusion,
    AverageFusion,
    SoftmaxFusion,
    ConfidenceGatedFusion,
    RegionAdaptiveFusion,
    UQFusionModule,
    create_fusion_module
)

from .learnable_fusion import (
    LearnableFusionConfig,
    LearnableFusionNetwork,
    AttentionFusionNetwork,
    UNetFusionNetwork,
    HybridFusion,
    create_learnable_fusion
)

from .losses import (
    FusionReconstructionLoss,
    WeightRegularizationLoss,
    UncertaintyConsistencyLoss,
    PerceptualFusionLoss,
    SSIMFusionLoss,
    CompositeFusionLoss,
    FusionQualityMetrics
)

__all__ = [
    # Fusion Module
    'FusionConfig',
    'UncertaintyGuidedFusion',
    'AverageFusion',
    'SoftmaxFusion',
    'ConfidenceGatedFusion',
    'RegionAdaptiveFusion',
    'UQFusionModule',
    'create_fusion_module',
    
    # Learnable Fusion
    'LearnableFusionConfig',
    'LearnableFusionNetwork',
    'AttentionFusionNetwork',
    'UNetFusionNetwork',
    'HybridFusion',
    'create_learnable_fusion',
    
    # Losses
    'FusionReconstructionLoss',
    'WeightRegularizationLoss',
    'UncertaintyConsistencyLoss',
    'PerceptualFusionLoss',
    'SSIMFusionLoss',
    'CompositeFusionLoss',
    'FusionQualityMetrics'
]
