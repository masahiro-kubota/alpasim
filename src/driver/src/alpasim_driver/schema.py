"""Configuration schema for VAM driver."""

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """VAM model configuration."""

    checkpoint_path: str = MISSING  # Path to VAM checkpoint
    device: str = MISSING  # Device to run inference on (cuda/cpu)
    dtype: str = MISSING  # Data type for inference (float16/float32)
    tokenizer_path: str = MISSING  # Path to JIT compiled VQ tokenizer


@dataclass
class InferenceConfig:
    """Inference configuration."""

    context_length: int = MISSING  # Number of temporal frames to use as context
    image_height: int = MISSING  # Expected image height (VAM scales frames to 288 px)
    image_width: int = MISSING  # Expected image width (VAM scales frames to 512 px)
    use_cameras: list[str] = MISSING  # List of cameras to use
    max_batch_size: int = MISSING  # Maximum batch size for inference


@dataclass
class RouteConfig:
    """Route and command configuration."""

    default_command: int = 2  # Default command: 0=right, 1=left, 2=straight
    use_waypoint_commands: bool = True  # Whether to interpret waypoints as commands
    command_distance_threshold: float = (
        2.0  # Lateral displacement threshold for command determination (meters)
    )
    min_lookahead_distance: float = (
        5.0  # Minimum distance to look ahead for waypoints (meters)
    )


@dataclass
class TrajectoryConfig:
    """Trajectory generation configuration."""

    prediction_horizon: int = MISSING  # Number of future points to predict (@ 2Hz)
    frequency_hz: int = MISSING  # Output frequency in Hz


@dataclass
class RectificationTargetConfig:
    """Target pinhole parameters for rectifying a rendered camera."""

    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    resolution_hw: tuple[int, int]
    radial: tuple[float, ...] = ()
    tangential: tuple[float, ...] = ()
    thin_prism: tuple[float, ...] = ()


@dataclass
class VAMDriverConfig:
    """Main VAM driver configuration."""

    # Model configuration
    model: ModelConfig = MISSING

    # Server configuration
    host: str = MISSING
    port: int = MISSING

    # Inference configuration
    inference: InferenceConfig = MISSING

    # Route configuration
    route: Optional[RouteConfig] = None

    # Trajectory configuration
    trajectory: TrajectoryConfig = MISSING

    # Output configuration
    output_dir: str = MISSING

    # If true, generates debug images in `output_dir`
    plot_debug_images: bool = False

    # Optional per-camera rectification definitions
    rectification: Optional[dict[str, RectificationTargetConfig]] = None
