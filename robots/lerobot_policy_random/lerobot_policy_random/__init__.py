# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_random import RandomPolicyConfig
from .modeling_random import RandomPolicy
from .processor_random import make_random_pre_post_processors

__all__ = [
    "RandomPolicyConfig",
    "RandomPolicy",
    "make_random_pre_post_processors",
]