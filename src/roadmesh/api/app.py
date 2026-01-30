"""
FastAPI Application

REST API for road mesh extraction from satellite imagery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch

from roadmesh.core.config import BBox, RoadMeshConfig, load_config
from roadmesh.data.tile_fetcher import TileFetcher
from roadmesh.models.architectures import create_model
from roadmesh.geometry.vectorizer import mask_to_mesh, Vectorizer, MeshBuilder


# Request/Response schemas
class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="Bounding box as (minx, miny, maxx, maxy) in WGS84"
    )
    zoom: int = Field(18, ge=10, le=20, description="Zoom level for tiles")
    output_format: Literal["geojson", "mesh", "both"] = Field(
        "both",
        description="Output format"
    )
    simplify_tolerance: float = Field(
        1.5,
        ge=0,
        description="Douglas-Peucker simplification tolerance"
    )
    min_area: float = Field(
        50.0,
        ge=0,
        description="Minimum polygon area to include"
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    geojson: Optional[dict] = None
    mesh: Optional[dict] = None
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    device: str
    version: str


# Application state
class AppState:
    """Global application state."""
    
    def __init__(self):
        self.config: Optional[RoadMeshConfig] = None
        self.model: Optional[torch.nn.Module] = None
        self.tile_fetcher: Optional[TileFetcher] = None
        self.device: str = "cpu"
        self.ready: bool = False


app_state = AppState()


def create_app(config_path: Optional[Path] = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="RoadMesh API",
        description="ML-powered road mesh extraction from satellite imagery",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup():
        """Initialize model and resources on startup."""
        # Load configuration
        app_state.config = load_config(config_path)
        
        # Set device
        app_state.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize tile fetcher
        app_state.tile_fetcher = TileFetcher(
            provider=app_state.config.data.tile_provider,
            cache_dir=app_state.config.data.cache_dir / "tiles",
        )
        
        # Load model if checkpoint exists
        checkpoint_path = app_state.config.model.checkpoint_path
        if checkpoint_path and Path(checkpoint_path).exists():
            app_state.model = create_model(
                architecture=app_state.config.model.architecture,
                num_classes=app_state.config.model.num_classes,
                pretrained=False,
                checkpoint_path=str(checkpoint_path),
                device=app_state.device,
            )
            app_state.model.eval()
            app_state.ready = True
            print(f"Model loaded from {checkpoint_path}")
        else:
            print("Warning: No model checkpoint found. Prediction endpoint disabled.")
    
    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        if app_state.tile_fetcher:
            await app_state.tile_fetcher.close()
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API health and model status."""
        return HealthResponse(
            status="healthy" if app_state.ready else "model_not_loaded",
            model_loaded=app_state.model is not None,
            device=app_state.device,
            version="0.1.0",
        )
    
    # Prediction endpoint
    @app.post("/api/v1/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """
        Extract road mesh from satellite imagery.
        
        Args:
            request: Prediction request with bbox and options
            
        Returns:
            GeoJSON and/or mesh data
        """
        if not app_state.ready:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please configure checkpoint_path."
            )
        
        # Validate bbox
        bbox = BBox(
            minx=request.bbox[0],
            miny=request.bbox[1],
            maxx=request.bbox[2],
            maxy=request.bbox[3],
        )
        
        # Check bbox size
        area = (bbox.maxx - bbox.minx) * (bbox.maxy - bbox.miny)
        max_area = app_state.config.api.max_bbox_area
        if area > max_area:
            raise HTTPException(
                status_code=400,
                detail=f"Bbox area ({area:.6f}) exceeds maximum ({max_area:.6f})"
            )
        
        try:
            # Fetch satellite imagery
            image, metadata = await app_state.tile_fetcher.fetch_bbox_async(
                bbox, request.zoom
            )
            
            # Preprocess image
            image_tensor = preprocess_image(image, app_state.device)
            
            # Run inference
            with torch.no_grad():
                if app_state.config.training.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = app_state.model(image_tensor)
                else:
                    output = app_state.model(image_tensor)
                
                # Get prediction mask
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                mask = (pred > 0.5).astype(np.uint8)
            
            # Convert to mesh/geojson
            result = mask_to_mesh(
                mask,
                bbox,
                graph=None,  # TODO: Add graph binding
                config=app_state.config.geometry,
            )
            
            # Build response
            response = PredictResponse(
                metadata={
                    "bbox": bbox.tuple,
                    "zoom": request.zoom,
                    "image_size": image.shape[:2],
                    "polygon_count": result["metadata"]["polygon_count"],
                }
            )
            
            if request.output_format in ["geojson", "both"]:
                response.geojson = result["geojson"]
            
            if request.output_format in ["mesh", "both"]:
                response.mesh = result["mesh"]
            
            return response
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    # Tile proxy endpoint (for caching)
    @app.get("/api/v1/tiles/{z}/{x}/{y}")
    async def get_tile(z: int, x: int, y: int):
        """
        Proxy satellite tile with caching.
        
        Args:
            z, x, y: Tile coordinates
            
        Returns:
            PNG image bytes
        """
        from fastapi.responses import Response
        
        try:
            tile = await app_state.tile_fetcher.fetch_tile_async(x, y, z)
            
            # Convert to PNG
            from PIL import Image
            import io
            
            img = Image.fromarray(tile)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            
            return Response(
                content=buffer.getvalue(),
                media_type="image/png",
                headers={"Cache-Control": "max-age=86400"}
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Tile not found: {str(e)}"
            )
    
    return app


def preprocess_image(image: np.ndarray, device: str) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: RGB numpy array (H, W, 3)
        device: Target device
        
    Returns:
        Normalized tensor (1, 3, H, W)
    """
    import cv2
    
    # Resize to model input size (512x512)
    target_size = 512
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor (B, C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device)


# Default app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    config_path: Optional[str] = None
):
    """
    Run the API server.
    
    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload for development
        config_path: Optional config file path
    """
    import uvicorn
    
    # Set config path in environment
    import os
    if config_path:
        os.environ["ROADMESH_CONFIG_PATH"] = config_path
    
    uvicorn.run(
        "roadmesh.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
