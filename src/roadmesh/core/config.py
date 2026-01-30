"""
RoadMesh Core Configuration

Pydantic-based configuration management with YAML support.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BBox(BaseModel):
    """Geographic bounding box."""
    
    minx: float = Field(..., description="Minimum longitude")
    miny: float = Field(..., description="Minimum latitude")
    maxx: float = Field(..., description="Maximum longitude")
    maxy: float = Field(..., description="Maximum latitude")
    
    @property
    def tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)
    
    def contains(self, lon: float, lat: float) -> bool:
        return self.minx <= lon <= self.maxx and self.miny <= lat <= self.maxy


class AugmentationConfig(BaseModel):
    """Data augmentation settings."""
    
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_90: bool = True
    rotate_limit: int = 15
    brightness_contrast: bool = True
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    blur_limit: int = 3
    noise: bool = True


class DataConfig(BaseModel):
    """Data pipeline configuration."""
    
    tile_size: int = Field(512, description="Input tile size in pixels")
    tile_provider: Literal["esri", "mapbox", "bing", "google"] = "esri"
    cache_dir: Path = Path("./data/cache")
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    
    default_zoom: int = 18
    default_bbox: Optional[BBox] = None
    
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)
    
    @field_validator("cache_dir", "raw_dir", "processed_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    
    architecture: Literal["dlinknet34", "unet_resnet34", "unet_efficientnet"] = "dlinknet34"
    input_size: int = 512
    num_classes: int = 1
    pretrained_backbone: bool = True
    dropout: float = 0.2
    
    # For loading trained weights
    checkpoint_path: Optional[Path] = None


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    
    batch_size: int = Field(4, description="Batch size (4 for 8GB VRAM)")
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Memory optimization for 8GB VRAM
    mixed_precision: bool = True
    gradient_accumulation: int = 2
    gradient_checkpointing: bool = False
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Learning rate scheduler
    scheduler_type: Literal["cosine", "step", "plateau"] = "cosine"
    warmup_epochs: int = 5
    
    # Loss weights
    bce_weight: float = 0.5
    dice_weight: float = 0.3
    connectivity_weight: float = 0.2
    
    # Checkpointing
    checkpoint_dir: Path = Path("./checkpoints")
    save_top_k: int = 3


class InferenceConfig(BaseModel):
    """Inference settings."""
    
    batch_size: int = 8
    tile_overlap: int = 64
    confidence_threshold: float = 0.5
    use_tta: bool = False  # Test-time augmentation


class GeometryConfig(BaseModel):
    """Vectorization and mesh settings."""
    
    simplify_tolerance: float = 1.5
    min_polygon_area: float = 50.0
    snap_tolerance: float = 2.0
    buffer_distance: float = 0.0
    
    # CRS settings
    default_crs: str = "EPSG:4326"
    working_crs: str = "EPSG:3857"  # Web Mercator for calculations


class APIConfig(BaseModel):
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    tile_cache_size: int = 1000
    request_timeout: int = 60
    max_bbox_area: float = 0.01  # Max area in degrees² per request
    
    cors_origins: list[str] = ["*"]
    api_key_required: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "console"
    log_file: Optional[Path] = None


class RoadMeshConfig(BaseModel):
    """Root configuration for RoadMesh."""
    
    project_name: str = "roadmesh"
    version: str = "0.1.0"
    
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "RoadMeshConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def merge_with(self, overrides: dict) -> "RoadMeshConfig":
        """Create new config with overrides applied."""
        current = self.model_dump()
        self._deep_merge(current, overrides)
        return RoadMeshConfig(**current)
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Deep merge override into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                RoadMeshConfig._deep_merge(base[key], value)
            else:
                base[key] = value


class Settings(BaseSettings):
    """Environment-based settings (secrets, paths)."""
    
    model_config = SettingsConfigDict(
        env_prefix="ROADMESH_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # Paths
    config_path: Path = Path("configs/base.yaml")
    data_dir: Path = Path("./data")
    weights_dir: Path = Path("./weights")
    
    # API keys (optional)
    mapbox_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # Device settings
    cuda_device: int = 0
    
    @property
    def device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{self.cuda_device}"
        return "cpu"


# Global settings instance
settings = Settings()


def load_config(config_path: Optional[Path] = None) -> RoadMeshConfig:
    """Load configuration from file or return default."""
    path = config_path or settings.config_path
    
    if path.exists():
        return RoadMeshConfig.from_yaml(path)
    
    return RoadMeshConfig()
