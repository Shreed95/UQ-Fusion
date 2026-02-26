# models/uncertainty/__init__.py

from .aleatoric import (
    AleatoricConfig,
    VarianceHead,
    HeteroscedasticLoss,
    AleatoricUncertaintyEstimator,
    DiffusionWithAleatoric,
    GANGeneratorWithAleatoric,
    AleatoricLoss,
    compute_aleatoric_uncertainty
)

from .epistemic import (
    EpistemicConfig,
    MCDropout,
    MCDropoutWrapper,
    EpistemicUncertaintyEstimator,
    DeepEnsemble,
    DiffusionEpistemicEstimator,
    GANEpistemicEstimator,
    compute_epistemic_uncertainty,
    entropy_from_predictions
)

from .combined import (
    CombinedUncertaintyConfig,
    CombinedUncertaintyEstimator,
    BranchUncertaintyEstimator,
    DualBranchUncertaintyEstimator,
    combine_uncertainties,
    uncertainty_quality_score
)

from .wrappers import (
    UncertaintyWrapperConfig,
    UncertaintyAwareDiffusion,
    UncertaintyAwareGAN,
    UncertaintyAwareDualBranch,
    load_uncertainty_aware_models
)

__all__ = [
    # Aleatoric
    'AleatoricConfig',
    'VarianceHead',
    'HeteroscedasticLoss',
    'AleatoricUncertaintyEstimator',
    'DiffusionWithAleatoric',
    'GANGeneratorWithAleatoric',
    'AleatoricLoss',
    'compute_aleatoric_uncertainty',
    
    # Epistemic
    'EpistemicConfig',
    'MCDropout',
    'MCDropoutWrapper',
    'EpistemicUncertaintyEstimator',
    'DeepEnsemble',
    'DiffusionEpistemicEstimator',
    'GANEpistemicEstimator',
    'compute_epistemic_uncertainty',
    'entropy_from_predictions',
    
    # Combined
    'CombinedUncertaintyConfig',
    'CombinedUncertaintyEstimator',
    'BranchUncertaintyEstimator',
    'DualBranchUncertaintyEstimator',
    'combine_uncertainties',
    'uncertainty_quality_score',
    
    # Wrappers
    'UncertaintyWrapperConfig',
    'UncertaintyAwareDiffusion',
    'UncertaintyAwareGAN',
    'UncertaintyAwareDualBranch',
    'load_uncertainty_aware_models'
]
