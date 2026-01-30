"""
RoadMesh - ML pipeline for road mesh extraction from satellite imagery.

Usage:
    from roadmesh import create_model, Trainer, TileFetcher
    
    # Create model
    model = create_model("dlinknet34", pretrained=True)
    
    # Train
    trainer = Trainer(model, config)
    trainer.fit(train_loader, val_loader)
"""
from roadmesh.core.config import (
    RoadMeshConfig,
    BBox,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from roadmesh.models.architectures import create_model, DLinkNet34
from roadmesh.data.tile_fetcher import TileFetcher
from roadmesh.geometry.vectorizer import Vectorizer, MeshBuilder, mask_to_mesh

__version__ = "0.1.0"
__all__ = [
    # Config
    "RoadMeshConfig",
    "BBox",
    "DataConfig", 
    "ModelConfig",
    "TrainingConfig",
    "load_config",
    # Models
    "create_model",
    "DLinkNet34",
    # Data
    "TileFetcher",
    # Geometry
    "Vectorizer",
    "MeshBuilder",
    "mask_to_mesh",
]
