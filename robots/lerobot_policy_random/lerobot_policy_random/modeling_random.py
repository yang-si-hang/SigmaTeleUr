# modeling_my_custom_policy.py
import torch
import torch.nn as nn
from typing import Dict, Any

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_random import RandomPolicyConfig

class RandomPolicy(PreTrainedPolicy):
    config_class = RandomPolicyConfig
    name = "random"

    def __init__(self, config: RandomPolicyConfig, dataset_stats: Dict[str, Any] = None):
        super().__init__(config, dataset_stats)
        config.validate_features()

        self.config = config
        self.action_dim = sum(v["shape"][0] for v in config.output_features.values())
        self.chunk_size = config.chunk_size if hasattr(config, "chunk_size") else 1

        self.reset()

    def reset(self) -> None:
        """Reset any internal state of the policy."""
        pass

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        """Select an action given the current observation batch."""
        batch_size = batch['observation.state'].shape[0]
        action_dim = self.config.output_features['action'].shape[0]
        random_actions = torch.rand(batch_size, action_dim, device=batch['observation.state'].device)
        return random_actions
    
    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        """Predict a chunk of actions given the current observation batch."""
        batch_size = batch['observation.state'].shape[0]
        chunk_size = self.config.chunk_size
        action_dim = self.config.output_features['action'].shape[0]
        random_action_chunks = torch.rand(batch_size, chunk_size, action_dim, device=batch['observation.state'].device)
        return random_action_chunks
    
    def forward(self, batch, **kwargs):
        """Forward pass of the policy network."""
        raise NotImplementedError("RandomPolicy does not implement a forward pass.")