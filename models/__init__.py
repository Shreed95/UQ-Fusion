# models/__init__.py

from .vae import (
    VAE,
    VAESmall,
    VAEWithLoss,
    VAEConfig,
    VAEEncoder,
    VAEDecoder,
    VAELoss,
    CombinedVAELoss,
    create_vae
)

from .diffusion import (
    DiffusionUNet,
    DiffusionUNetSmall,
    UNetConfig,
    create_unet,
    NoiseScheduler,
    DDPMScheduler,
    DDIMScheduler,
    SchedulerConfig,
    DDPMSampler,
    DDIMSampler,
    GuidedSampler,
    ImageToImageSampler,
    LatentDiffusionModel,
    LatentDiffusionModelSmall,
    LatentDiffusionConfig,
    create_latent_diffusion
)

from .gan import (
    STABLEGenerator,
    STABLEGeneratorWithSkip,
    STABLEGeneratorSmall,
    GeneratorConfig,
    PatchGANDiscriminator,
    PatchGANDiscriminatorSmall,
    MultiScaleDiscriminator,
    ConditionalPatchGAN,
    DiscriminatorConfig,
    create_generator,
    create_discriminator,
    AdversarialLoss,
    L1Loss,
    L2Loss,
    SSIMLoss,
    GradientLoss,
    SpatialPreservationLoss,
    QuantitativePreservationLoss,
    IdentityLoss,
    PerceptualLoss,
    STABLEGANLoss
)

from .uncertainty import (
    # Aleatoric
    AleatoricConfig,
    VarianceHead,
    HeteroscedasticLoss,
    AleatoricUncertaintyEstimator,
    DiffusionWithAleatoric,
    GANGeneratorWithAleatoric,
    AleatoricLoss,
    compute_aleatoric_uncertainty,
    
    # Epistemic
    EpistemicConfig,
    MCDropout,
    MCDropoutWrapper,
    EpistemicUncertaintyEstimator,
    DeepEnsemble,
    DiffusionEpistemicEstimator,
    GANEpistemicEstimator,
    compute_epistemic_uncertainty,
    entropy_from_predictions,
    
    # Combined
    CombinedUncertaintyConfig,
    CombinedUncertaintyEstimator,
    BranchUncertaintyEstimator,
    DualBranchUncertaintyEstimator,
    combine_uncertainties,
    uncertainty_quality_score,
    
    # Wrappers
    UncertaintyWrapperConfig,
    UncertaintyAwareDiffusion,
    UncertaintyAwareGAN,
    UncertaintyAwareDualBranch,
    load_uncertainty_aware_models
)

from .fusion import (
    # Fusion Module
    FusionConfig,
    UncertaintyGuidedFusion,
    AverageFusion,
    SoftmaxFusion,
    ConfidenceGatedFusion,
    RegionAdaptiveFusion,
    UQFusionModule,
    create_fusion_module,
    
    # Learnable Fusion
    LearnableFusionConfig,
    LearnableFusionNetwork,
    AttentionFusionNetwork,
    UNetFusionNetwork,
    HybridFusion,
    create_learnable_fusion,
    
    # Losses
    FusionReconstructionLoss,
    WeightRegularizationLoss,
    UncertaintyConsistencyLoss,
    PerceptualFusionLoss,
    SSIMFusionLoss,
    CompositeFusionLoss,
    FusionQualityMetrics
)

from .segmentation import (
    # U-Net
    UNetConfig,
    SegmentationUNet,
    SegmentationUNetSmall,
    SegmentationUNetLarge,
    create_segmentation_model,
    
    # Losses
    DiceLoss,
    GeneralizedDiceLoss,
    FocalLoss,
    TverskyLoss,
    CombinedSegmentationLoss,
    BraTSLoss,
    
    # Metrics
    BraTSRegions,
    DiceScore,
    HausdorffDistance,
    IoUScore,
    SensitivitySpecificity,
    SegmentationMetrics,
    compute_dice_score,
    compute_segmentation_metrics
)

__all__ = [
    # VAE
    'VAE', 'VAESmall', 'VAEWithLoss', 'VAEConfig', 'VAEEncoder', 'VAEDecoder',
    'VAELoss', 'CombinedVAELoss', 'create_vae',
    
    # Diffusion
    'DiffusionUNet', 'DiffusionUNetSmall', 'UNetConfig', 'create_unet',
    'NoiseScheduler', 'DDPMScheduler', 'DDIMScheduler', 'SchedulerConfig',
    'DDPMSampler', 'DDIMSampler', 'GuidedSampler', 'ImageToImageSampler',
    'LatentDiffusionModel', 'LatentDiffusionModelSmall', 'LatentDiffusionConfig',
    'create_latent_diffusion',
    
    # GAN
    'STABLEGenerator', 'STABLEGeneratorWithSkip', 'STABLEGeneratorSmall',
    'GeneratorConfig', 'create_generator',
    'PatchGANDiscriminator', 'PatchGANDiscriminatorSmall', 'MultiScaleDiscriminator',
    'ConditionalPatchGAN', 'DiscriminatorConfig', 'create_discriminator',
    'AdversarialLoss', 'L1Loss', 'L2Loss', 'SSIMLoss', 'GradientLoss',
    'SpatialPreservationLoss', 'QuantitativePreservationLoss', 'IdentityLoss',
    'PerceptualLoss', 'STABLEGANLoss',
    
    # Uncertainty
    'AleatoricConfig', 'VarianceHead', 'HeteroscedasticLoss',
    'AleatoricUncertaintyEstimator', 'DiffusionWithAleatoric',
    'GANGeneratorWithAleatoric', 'AleatoricLoss', 'compute_aleatoric_uncertainty',
    'EpistemicConfig', 'MCDropout', 'MCDropoutWrapper', 'EpistemicUncertaintyEstimator',
    'DeepEnsemble', 'DiffusionEpistemicEstimator', 'GANEpistemicEstimator',
    'compute_epistemic_uncertainty', 'entropy_from_predictions',
    'CombinedUncertaintyConfig', 'CombinedUncertaintyEstimator',
    'BranchUncertaintyEstimator', 'DualBranchUncertaintyEstimator',
    'combine_uncertainties', 'uncertainty_quality_score',
    'UncertaintyWrapperConfig', 'UncertaintyAwareDiffusion', 'UncertaintyAwareGAN',
    'UncertaintyAwareDualBranch', 'load_uncertainty_aware_models',
    
    # Fusion
    'FusionConfig', 'UncertaintyGuidedFusion', 'AverageFusion', 'SoftmaxFusion',
    'ConfidenceGatedFusion', 'RegionAdaptiveFusion', 'UQFusionModule',
    'create_fusion_module',
    'LearnableFusionConfig', 'LearnableFusionNetwork', 'AttentionFusionNetwork',
    'UNetFusionNetwork', 'HybridFusion', 'create_learnable_fusion',
    'FusionReconstructionLoss', 'WeightRegularizationLoss', 'UncertaintyConsistencyLoss',
    'PerceptualFusionLoss', 'SSIMFusionLoss', 'CompositeFusionLoss', 'FusionQualityMetrics',
    
    # Segmentation
    'UNetConfig', 'SegmentationUNet', 'SegmentationUNetSmall', 'SegmentationUNetLarge',
    'create_segmentation_model',
    'DiceLoss', 'GeneralizedDiceLoss', 'FocalLoss', 'TverskyLoss',
    'CombinedSegmentationLoss', 'BraTSLoss',
    'BraTSRegions', 'DiceScore', 'HausdorffDistance', 'IoUScore',
    'SensitivitySpecificity', 'SegmentationMetrics',
    'compute_dice_score', 'compute_segmentation_metrics'
]
