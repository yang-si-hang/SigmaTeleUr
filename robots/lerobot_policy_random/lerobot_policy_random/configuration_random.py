# configuration_my_custom_policy.py
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

@PreTrainedConfig.register_subclass("random")
@dataclass
class RandomPolicyConfig(PreTrainedConfig):
    """Configuration class for MyCustomPolicy.

    Args:
        n_obs_steps: Number of observation steps to use as input
        horizon: Action prediction horizon
        n_action_steps: Number of action steps to execute
        hidden_dim: Hidden dimension for the policy network
        # Add your policy-specific parameters here
    """
    # ...PreTrainedConfig fields...
    pass

    def __post_init__(self):
        super().__post_init__()
        # Add any validation logic here

    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
