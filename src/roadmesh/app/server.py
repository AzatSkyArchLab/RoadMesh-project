"""
RoadMesh Interactive Server

Web application for training and inference with live visualization.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import box

# Create app
app = FastAPI(title="RoadMesh", version="0.1.0")


# Available GeoJSON sources
GEOJSON_SOURCES = {
    "osi_sush": {
        "name": "ОСИ Москвы (официальная)",
        "path": "data/geojson_data/osi_sush.geojson",
        "road_type_field": "kl_gp",
        "description": "Официальная дорожная сеть Москвы"
    },
    "osm": {
        "name": "OpenStreetMap",
        "path": "data/geojson_data/osm_roads.geojson",
        "road_type_field": "highway",
        "description": "Дороги из OpenStreetMap"
    },
}


# Global state
class AppState:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.jobs: dict[str, dict] = {}
        self.models: dict[str, Path] = {}
        self.current_source: str = "osi_sush"  # Default source
        self.custom_geojson: dict = None  # Custom uploaded GeoJSON

state = AppState()


# Request/Response models
class BBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float


class TrainRequest(BaseModel):
    bbox: BBox
    name: Optional[str] = None
    epochs: int = 50


class PredictRequest(BaseModel):
    bbox: BBox
    model_name: Optional[str] = None
    bind_to_graph: bool = True  # Привязывать детекции к дорожному графу
    source: Optional[str] = None  # Источник данных для привязки


# WebSocket for progress updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.active_connections:
            state.active_connections.remove(websocket)


async def broadcast(message: dict):
    """Send message to all connected clients."""
    for connection in state.active_connections:
        try:
            await connection.send_json(message)
        except:
            pass


# API Endpoints
@app.post("/api/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start training job."""
    job_id = str(uuid.uuid4())[:8]
    name = request.name or f"model_{job_id}"
    
    state.jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Starting...",
        "name": name,
        "bbox": request.bbox.model_dump(),
    }
    
    background_tasks.add_task(run_training, job_id, request)
    return {"job_id": job_id, "status": "started"}


@app.post("/api/predict")
async def run_prediction(request: PredictRequest, background_tasks: BackgroundTasks):
    """Run prediction on area."""
    job_id = str(uuid.uuid4())[:8]
    
    state.jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Starting prediction...",
    }
    
    background_tasks.add_task(run_inference, job_id, request)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in state.jobs:
        return {"error": "Job not found"}
    return state.jobs[job_id]


@app.get("/api/models")
async def list_models():
    """List available trained models."""
    models_dir = Path("checkpoints")
    models = []
    if models_dir.exists():
        for f in models_dir.glob("*.pt"):
            models.append({
                "name": f.stem,
                "path": str(f),
                "size_mb": f.stat().st_size / 1024 / 1024,
                "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    return {"models": models}


@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a trained model."""
    models_dir = Path("checkpoints")
    model_path = models_dir / f"{model_name}.pt"

    if not model_path.exists():
        return {"error": f"Model not found: {model_name}"}

    try:
        model_path.unlink()
        return {"status": "ok", "message": f"Deleted {model_name}"}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/models")
async def delete_all_models():
    """Delete all trained models."""
    models_dir = Path("checkpoints")
    deleted = []

    if models_dir.exists():
        for f in models_dir.glob("*.pt"):
            try:
                f.unlink()
                deleted.append(f.stem)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

    return {"status": "ok", "deleted": deleted, "count": len(deleted)}


@app.post("/api/models/{model_name}/activate")
async def activate_model(model_name: str):
    """Set a model as the active (best) model for inference."""
    models_dir = Path("checkpoints")
    source_path = models_dir / f"{model_name}.pt"
    best_path = models_dir / "best_model.pt"

    if not source_path.exists():
        return {"error": f"Model not found: {model_name}"}

    if model_name == "best_model":
        return {"status": "ok", "message": "Already the active model"}

    try:
        import shutil
        # Backup current best if exists
        if best_path.exists():
            backup_name = f"best_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            shutil.copy(best_path, models_dir / backup_name)

        # Copy selected model to best_model
        shutil.copy(source_path, best_path)
        return {"status": "ok", "message": f"Activated {model_name} as best_model"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sources")
async def list_sources():
    """List available GeoJSON data sources."""
    sources = []
    for key, info in GEOJSON_SOURCES.items():
        path = Path(info["path"])
        sources.append({
            "id": key,
            "name": info["name"],
            "description": info["description"],
            "exists": path.exists(),
            "road_type_field": info["road_type_field"],
        })

    # Add custom source if uploaded
    if state.custom_geojson:
        sources.append({
            "id": "custom",
            "name": "Пользовательский GeoJSON",
            "description": f"{len(state.custom_geojson.get('features', []))} объектов",
            "exists": True,
            "road_type_field": "auto",
        })

    return {
        "sources": sources,
        "current": state.current_source,
    }


@app.post("/api/sources/{source_id}")
async def set_source(source_id: str):
    """Set active GeoJSON source."""
    if source_id not in GEOJSON_SOURCES and source_id != "custom":
        return {"error": f"Unknown source: {source_id}"}

    if source_id == "custom" and not state.custom_geojson:
        return {"error": "No custom GeoJSON uploaded"}

    state.current_source = source_id
    return {"status": "ok", "current": source_id}


@app.post("/api/sources/upload")
async def upload_geojson(file: UploadFile = File(...)):
    """Upload custom GeoJSON file."""
    try:
        content = await file.read()
        geojson = json.loads(content.decode('utf-8'))

        if geojson.get("type") != "FeatureCollection":
            return {"error": "Invalid GeoJSON: must be FeatureCollection"}

        features = geojson.get("features", [])
        if not features:
            return {"error": "GeoJSON has no features"}

        # Detect properties
        props = features[0].get("properties", {})
        geom_type = features[0].get("geometry", {}).get("type", "Unknown")

        state.custom_geojson = geojson
        state.current_source = "custom"

        # Save to file for persistence
        custom_path = Path("data/geojson_data/custom_upload.geojson")
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        with open(custom_path, "w") as f:
            json.dump(geojson, f)

        return {
            "status": "ok",
            "features_count": len(features),
            "geometry_type": geom_type,
            "properties": list(props.keys())[:15],
            "sample_properties": props,
        }
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/ground-truth")
async def get_ground_truth(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    source: Optional[str] = None
):
    """Get ground truth roads for bbox from selected source."""
    # Use specified source or current default
    source_id = source or state.current_source

    # Get GeoJSON data
    if source_id == "custom":
        if not state.custom_geojson:
            return {"type": "FeatureCollection", "features": [], "source": "custom"}
        # Filter custom GeoJSON by bbox
        gdf = gpd.GeoDataFrame.from_features(state.custom_geojson["features"])
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
    elif source_id in GEOJSON_SOURCES:
        vector_path = Path(GEOJSON_SOURCES[source_id]["path"])
        if not vector_path.exists():
            return {"type": "FeatureCollection", "features": [], "source": source_id}
        gdf = gpd.read_file(vector_path, bbox=(minx, miny, maxx, maxy))
    else:
        return {"type": "FeatureCollection", "features": [], "error": "Unknown source"}

    if gdf.empty:
        return {"type": "FeatureCollection", "features": [], "source": source_id}

    bbox_geom = box(minx, miny, maxx, maxy)
    gdf = gdf[gdf.intersects(bbox_geom)]

    result = json.loads(gdf.to_json())
    result["source"] = source_id
    result["source_name"] = GEOJSON_SOURCES.get(source_id, {}).get("name", "Custom")

    return result


@app.get("/api/predictions/{job_id}")
async def get_predictions(job_id: str):
    """Get prediction results as GeoJSON."""
    pred_path = Path(f"data/predictions/{job_id}.geojson")
    
    if pred_path.exists():
        with open(pred_path) as f:
            return json.load(f)
    
    return {"type": "FeatureCollection", "features": []}


# Background tasks
async def run_training(job_id: str, request: TrainRequest):
    """Run training pipeline."""
    print(f"\n{'='*60}")
    print(f"[TRAIN] Starting job {job_id}")
    print(f"[TRAIN] BBox: {request.bbox}")
    print(f"[TRAIN] Epochs: {request.epochs}")
    print(f"{'='*60}\n")
    
    try:
        async def update(progress: float, message: str, status: str = "running"):
            print(f"[TRAIN] {progress:.0f}% - {message}")
            state.jobs[job_id].update({
                "status": status,
                "progress": progress,
                "message": message
            })
            await broadcast({"type": "progress", "job_id": job_id, **state.jobs[job_id]})
            await asyncio.sleep(0.1)
        
        await update(0, "Initializing...")
        
        from roadmesh.models.architectures import create_model
        from roadmesh.training.trainer import Trainer
        from roadmesh.core.config import TrainingConfig
        from roadmesh.data.dataset import create_dataloaders
        
        # Step 1: Use existing test dataset (bypass async issues)
        await update(5, "Checking dataset...")
        
        output_dir = Path("data/osm_dataset")
        
        if not output_dir.exists() or not (output_dir / "train" / "images").exists():
            await update(100, "No dataset found. Run: python scripts/prepare_dataset.py first", "failed")
            return
        
        train_count = len(list((output_dir / "train" / "images").glob("*.png")))
        val_count = len(list((output_dir / "val" / "images").glob("*.png")))
        
        print(f"[TRAIN] Using existing dataset: {train_count} train, {val_count} val")
        counts = {"train": train_count, "val": val_count}
        
        await update(40, f"Dataset ready: {counts['train']} train, {counts['val']} val")
        
        if counts['train'] < 2:
            await update(100, f"Not enough training data: only {counts['train']} samples", "failed")
            return
        
        # Step 2: Load model
        await update(45, "Loading model...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TRAIN] Using device: {device}")
        
        model = create_model("dlinknet34", pretrained=True, device=device)
        
        config = TrainingConfig(
            batch_size=2 if device == "cuda" else 1,
            epochs=request.epochs,
            learning_rate=0.0001,
            mixed_precision=(device == "cuda"),
            gradient_accumulation=2,
            checkpoint_dir=Path("checkpoints"),
            early_stopping_patience=10,
        )
        
        train_loader, val_loader = create_dataloaders(
            output_dir,
            batch_size=config.batch_size,
            num_workers=0  # Windows compatibility
        )
        
        print(f"[TRAIN] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        await update(50, "Training started...")
        
        trainer = Trainer(model, config, device, experiment_name=request.name or job_id)
        
        # Step 3: Training loop
        total_epochs = request.epochs
        for epoch in range(total_epochs):
            print(f"[TRAIN] Epoch {epoch+1}/{total_epochs}")
            
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            
            progress = 50 + (epoch + 1) / total_epochs * 45
            await update(
                progress,
                f"Epoch {epoch+1}/{total_epochs} - Loss: {train_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}"
            )
            
            trainer.current_epoch = epoch
            
            if val_metrics['iou'] > trainer.best_metric:
                trainer.best_metric = val_metrics['iou']
                trainer.save_checkpoint("best_model.pt", is_best=True)
                print(f"[TRAIN] New best IoU: {val_metrics['iou']:.4f}")
        
        # Step 4: Save final model
        model_path = Path(f"checkpoints/{request.name or job_id}_final.pt")
        trainer.save_checkpoint(model_path.name)
        state.models[request.name or job_id] = model_path
        
        await update(100, f"Training complete! Best IoU: {trainer.best_metric:.4f}", "completed")
        state.jobs[job_id]["model_path"] = str(model_path)
        
        print(f"[TRAIN] Completed! Model saved to {model_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[TRAIN] Error: {str(e)}")
        state.jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Error: {str(e)}"
        })
        await broadcast({"type": "progress", "job_id": job_id, **state.jobs[job_id]})


async def run_inference(job_id: str, request: PredictRequest):
    """Run inference on area with optional graph binding."""
    print(f"\n{'='*60}")
    print(f"[PREDICT] Starting job {job_id}")
    print(f"[PREDICT] BBox: {request.bbox}")
    print(f"[PREDICT] Bind to graph: {request.bind_to_graph}")
    print(f"{'='*60}\n")

    try:
        bbox = request.bbox

        async def update(progress: float, message: str, status: str = "running"):
            print(f"[PREDICT] {progress:.0f}% - {message}")
            state.jobs[job_id].update({
                "status": status,
                "progress": progress,
                "message": message
            })
            await broadcast({"type": "progress", "job_id": job_id, **state.jobs[job_id]})
            await asyncio.sleep(0.1)

        await update(0, "Loading model...")

        from roadmesh.core.config import BBox as BBoxConfig
        from roadmesh.data.tile_fetcher import TileFetcher
        from roadmesh.models.architectures import create_model
        from roadmesh.geometry.vectorizer import Vectorizer, GraphBinder, MeshBuilder

        model_path = Path("checkpoints/best_model.pt")
        if request.model_name:
            model_path = Path(f"checkpoints/{request.model_name}.pt")

        if not model_path.exists():
            await update(0, "No trained model found. Train first!", "failed")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[PREDICT] Using device: {device}")
        print(f"[PREDICT] Loading model from: {model_path}")

        model = create_model("dlinknet34", checkpoint_path=str(model_path), device=device)
        model = model.float()  # Ensure FP32 weights (checkpoint may have FP16 from mixed precision training)
        model.eval()

        await update(10, "Fetching satellite tiles...")

        bbox_config = BBoxConfig(minx=bbox.minx, miny=bbox.miny, maxx=bbox.maxx, maxy=bbox.maxy)
        fetcher = TileFetcher(provider="esri", cache_dir=Path("data/cache/tiles"))

        image, metadata = await fetcher.fetch_bbox_async(bbox_config, zoom=18)
        print(f"[PREDICT] Fetched image: {image.shape}")

        await update(40, "Running inference...")

        img_resized = cv2.resize(image, (512, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

        # Normalize on the same device with explicit float32
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Ensure input is float32 (not half precision)
        img_tensor = img_tensor.float()
        print(f"[PREDICT] Input tensor dtype: {img_tensor.dtype}, device: {img_tensor.device}")

        # Disable autocast and ensure float32 inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            img_tensor = img_tensor.float()  # Ensure float32 even inside context
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (pred > 0.5).astype(np.uint8)

        print(f"[PREDICT] Mask shape: {mask.shape}, road pixels: {mask.sum()}")

        await update(70, "Vectorizing results...")

        vectorizer = Vectorizer()
        polygons_px = vectorizer.mask_to_polygons(mask * 255)

        print(f"[PREDICT] Found {len(polygons_px)} polygons")

        properties_list = None
        graph_info = None

        if polygons_px:
            polygons_geo = vectorizer.polygons_to_geo(
                polygons_px,
                bbox_config,
                (mask.shape[1], mask.shape[0])
            )

            # Bind to road graph if requested
            if request.bind_to_graph and polygons_geo:
                await update(80, "Binding to road graph...")

                source_id = request.source or state.current_source
                graph_gdf = None

                # Load graph from selected source
                if source_id == "custom" and state.custom_geojson:
                    graph_gdf = gpd.GeoDataFrame.from_features(
                        state.custom_geojson["features"]
                    )
                    if graph_gdf.crs is None:
                        graph_gdf.set_crs("EPSG:4326", inplace=True)
                elif source_id in GEOJSON_SOURCES:
                    graph_path = Path(GEOJSON_SOURCES[source_id]["path"])
                    if graph_path.exists():
                        graph_gdf = gpd.read_file(
                            graph_path,
                            bbox=(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
                        )

                if graph_gdf is not None and not graph_gdf.empty:
                    print(f"[PREDICT] Binding to {len(graph_gdf)} graph edges")
                    binder = GraphBinder(graph=graph_gdf, snap_tolerance=10.0)
                    properties_list = binder.bind_polygons(polygons_geo)

                    # Count bound polygons
                    bound_count = sum(1 for p in properties_list if p.get("bound"))
                    graph_info = {
                        "source": source_id,
                        "source_name": GEOJSON_SOURCES.get(source_id, {}).get("name", "Custom"),
                        "edges_in_bbox": len(graph_gdf),
                        "bound_polygons": bound_count,
                        "unbound_polygons": len(polygons_geo) - bound_count,
                    }
                    print(f"[PREDICT] Bound {bound_count}/{len(polygons_geo)} polygons")

            geojson = vectorizer.to_geojson(polygons_geo, properties_list)

            # Build mesh for Three.js visualization
            await update(85, "Building mesh...")
            mesh_builder = MeshBuilder(default_elevation=0.5)
            mesh = mesh_builder.polygons_to_mesh_collection(
                polygons_geo,
                properties_list,
                elevation=0.5
            )
        else:
            geojson = {"type": "FeatureCollection", "features": []}
            mesh = {"type": "MeshCollection", "features": [], "count": 0}

        await update(90, "Saving results...")

        pred_dir = Path("data/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Save GeoJSON
        with open(pred_dir / f"{job_id}.geojson", "w") as f:
            json.dump(geojson, f)

        # Save Mesh
        with open(pred_dir / f"{job_id}_mesh.json", "w") as f:
            json.dump(mesh, f)

        result = {
            "geojson": geojson,
            "mesh": mesh,
            "polygon_count": len(geojson["features"]),
        }

        if graph_info:
            result["graph_binding"] = graph_info

        state.jobs[job_id]["result"] = result

        msg = f"Done! Found {len(geojson['features'])} road polygons"
        if graph_info:
            msg += f" ({graph_info['bound_polygons']} bound to {graph_info['source_name']})"

        await update(100, msg, "completed")

        await fetcher.close()

        print(f"[PREDICT] Completed! {msg}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PREDICT] Error: {str(e)}")
        state.jobs[job_id].update({
            "status": "failed",
            "message": f"Error: {str(e)}"
        })
        await broadcast({"type": "progress", "job_id": job_id, **state.jobs[job_id]})


# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>RoadMesh - Interactive Road Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        #map { position: absolute; top: 0; bottom: 0; left: 0; right: 300px; }
        #sidebar {
            position: absolute; top: 0; right: 0; width: 300px; height: 100%;
            background: #1a1a2e; color: white; padding: 20px; overflow-y: auto;
        }
        h1 { font-size: 1.5em; margin: 0 0 20px 0; color: #00d4ff; }
        h3 { font-size: 1em; margin: 15px 0 10px 0; color: #888; }
        .btn {
            display: block; width: 100%; padding: 12px; margin: 8px 0;
            border: none; border-radius: 6px; font-size: 14px; font-weight: 600;
            cursor: pointer; transition: all 0.2s;
        }
        .btn-primary { background: #00d4ff; color: #000; }
        .btn-primary:hover { background: #00b8e6; }
        .btn-primary:disabled { background: #444; color: #888; cursor: not-allowed; }
        .btn-success { background: #00ff88; color: #000; }
        .btn-success:hover { background: #00e077; }
        .btn-success:disabled { background: #444; color: #888; cursor: not-allowed; }
        .btn-danger { background: #ff4757; color: #fff; }
        .progress-container {
            background: #2d2d44; border-radius: 6px; padding: 15px; margin: 15px 0; display: none;
        }
        .progress-bar { height: 8px; background: #444; border-radius: 4px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #00ff88); width: 0%; transition: width 0.3s; }
        .progress-text { font-size: 12px; color: #888; }
        .status { padding: 8px 12px; border-radius: 4px; margin: 10px 0; font-size: 13px; }
        .status.info { background: #1e3a5f; color: #00d4ff; }
        .status.success { background: #1e5f3a; color: #00ff88; }
        .status.error { background: #5f1e1e; color: #ff4757; }
        .bbox-info { background: #2d2d44; padding: 10px; border-radius: 6px; font-size: 12px; margin: 10px 0; }
        .legend { background: #2d2d44; padding: 10px; border-radius: 6px; margin-top: 20px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; font-size: 12px; }
        .legend-color { width: 20px; height: 4px; margin-right: 10px; border-radius: 2px; }
        input[type="number"] {
            width: 100%; padding: 8px; border: 1px solid #444; border-radius: 4px;
            background: #2d2d44; color: white; margin: 5px 0;
        }
        label { font-size: 12px; color: #888; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="sidebar">
        <h1>🛣️ RoadMesh</h1>
        <div class="status info" id="status">Draw a rectangle on the map to select area</div>
        <div class="bbox-info" id="bbox-info" style="display:none">
            <strong>Selected Area:</strong><br><span id="bbox-text"></span>
        </div>
        <h3>1. SELECT AREA</h3>
        <button class="btn btn-primary" id="btn-draw">📍 Draw Rectangle</button>
        <button class="btn btn-danger" id="btn-clear" style="display:none">✕ Clear Selection</button>
        <h3>2. DATA SOURCE</h3>
        <select id="source-select" class="btn" style="background:#2d2d44;color:white;text-align:left;">
            <option value="osi_sush">ОСИ Москвы (официальная)</option>
            <option value="osm">OpenStreetMap</option>
            <option value="custom" disabled>Свой GeoJSON (загрузите)</option>
        </select>
        <input type="file" id="geojson-upload" accept=".geojson,.json" style="display:none">
        <button class="btn" style="background:#444" id="btn-upload">📁 Загрузить свой GeoJSON</button>
        <div id="upload-info" style="display:none;font-size:11px;color:#888;margin:5px 0;"></div>
        <h3>3. TRAIN MODEL</h3>
        <label>Epochs:</label>
        <input type="number" id="epochs" value="10" min="1" max="200">
        <button class="btn btn-primary" id="btn-train" disabled>🧠 Train Model</button>

        <h3>📦 MODELS</h3>
        <div id="models-list" style="background:#2d2d44;border-radius:6px;padding:10px;margin:10px 0;max-height:150px;overflow-y:auto;font-size:12px;">
            Loading...
        </div>
        <div style="display:flex;gap:5px;">
            <button class="btn" style="background:#444;flex:1;padding:8px;" id="btn-refresh-models">🔄 Refresh</button>
            <button class="btn btn-danger" style="flex:1;padding:8px;" id="btn-delete-all">🗑️ Delete All</button>
        </div>

        <h3>4. DETECT ROADS</h3>
        <label style="display:flex;align-items:center;margin:5px 0;">
            <input type="checkbox" id="bind-graph" checked style="margin-right:8px;">
            Привязать к дорожному графу
        </label>
        <button class="btn btn-success" id="btn-predict" disabled>🔍 Detect Roads</button>
        <div class="progress-container" id="progress">
            <div class="progress-text" id="progress-message">Processing...</div>
            <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
            <div class="progress-text"><span id="progress-percent">0</span>%</div>
        </div>
        <div class="legend">
            <strong>Legend</strong>
            <div class="legend-item"><div class="legend-color" style="background:#00ff00"></div>Ground Truth (source)</div>
            <div class="legend-item"><div class="legend-color" style="background:#ff6600"></div>Detected (bound)</div>
            <div class="legend-item"><div class="legend-color" style="background:#ff0000"></div>Detected (unbound)</div>
            <div class="legend-item"><div class="legend-color" style="background:#00d4ff"></div>Selected Area</div>
        </div>
    </div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <script>
        const map = L.map('map').setView([55.751244, 37.618423], 14);
        const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {attribution: 'Esri', maxZoom: 19}).addTo(map);
        const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution: 'OSM', maxZoom: 19});
        L.control.layers({"Satellite": satellite, "OpenStreetMap": osm}).addTo(map);

        const drawnItems = new L.FeatureGroup().addTo(map);
        const drawControl = new L.Control.Draw({
            draw: {rectangle: {shapeOptions: {color: '#00d4ff', weight: 3}}, polygon: false, polyline: false, circle: false, marker: false, circlemarker: false},
            edit: {featureGroup: drawnItems}
        });

        let gtLayer = null, predLayer = null, currentBbox = null;
        const btnDraw = document.getElementById('btn-draw');
        const btnClear = document.getElementById('btn-clear');
        const btnTrain = document.getElementById('btn-train');
        const btnPredict = document.getElementById('btn-predict');
        const statusEl = document.getElementById('status');
        const bboxInfo = document.getElementById('bbox-info');
        const bboxText = document.getElementById('bbox-text');
        const progressContainer = document.getElementById('progress');
        const progressFill = document.getElementById('progress-fill');
        const progressMessage = document.getElementById('progress-message');
        const progressPercent = document.getElementById('progress-percent');

        let ws = null;
        function connectWS() {
            ws = new WebSocket('ws://' + window.location.host + '/ws');
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                console.log('WS message:', data);
                if (data.type === 'progress') updateProgress(data);
            };
            ws.onclose = () => setTimeout(connectWS, 1000);
            ws.onopen = () => console.log('WebSocket connected');
        }
        connectWS();

        function setStatus(text, type) {
            statusEl.textContent = text;
            statusEl.className = 'status ' + (type || 'info');
        }

        function updateProgress(data) {
            progressFill.style.width = data.progress + '%';
            progressPercent.textContent = Math.round(data.progress);
            progressMessage.textContent = data.message;
            if (data.status === 'completed') {
                setStatus(data.message, 'success');
                progressContainer.style.display = 'none';
                if (data.result && data.result.geojson) showPredictions(data.result.geojson);
            } else if (data.status === 'failed') {
                setStatus(data.message, 'error');
                progressContainer.style.display = 'none';
            }
        }

        btnDraw.onclick = () => map.addControl(drawControl);

        map.on('draw:created', async (e) => {
            drawnItems.clearLayers();
            drawnItems.addLayer(e.layer);
            map.removeControl(drawControl);
            const bounds = e.layer.getBounds();
            currentBbox = {minx: bounds.getWest(), miny: bounds.getSouth(), maxx: bounds.getEast(), maxy: bounds.getNorth()};
            bboxText.textContent = currentBbox.minx.toFixed(4) + ', ' + currentBbox.miny.toFixed(4) + ' -> ' + currentBbox.maxx.toFixed(4) + ', ' + currentBbox.maxy.toFixed(4);
            bboxInfo.style.display = 'block';
            btnClear.style.display = 'block';
            btnTrain.disabled = false;
            btnPredict.disabled = false;
            setStatus('Loading ground truth...', 'info');
            await loadGroundTruth();
        });

        btnClear.onclick = () => {
            drawnItems.clearLayers();
            if (gtLayer) map.removeLayer(gtLayer);
            if (predLayer) map.removeLayer(predLayer);
            currentBbox = null;
            bboxInfo.style.display = 'none';
            btnClear.style.display = 'none';
            btnTrain.disabled = true;
            btnPredict.disabled = true;
            setStatus('Draw a rectangle on the map to select area', 'info');
        };

        async function loadGroundTruth() {
            if (!currentBbox) return;
            const source = document.getElementById('source-select').value;
            const url = '/api/ground-truth?minx=' + currentBbox.minx + '&miny=' + currentBbox.miny + '&maxx=' + currentBbox.maxx + '&maxy=' + currentBbox.maxy + '&source=' + source;
            const resp = await fetch(url);
            const geojson = await resp.json();
            if (gtLayer) map.removeLayer(gtLayer);
            gtLayer = L.geoJSON(geojson, {style: {color: '#00ff00', weight: 3, opacity: 0.8}}).addTo(map);
            const sourceName = geojson.source_name || source;
            setStatus('Loaded ' + geojson.features.length + ' road segments from ' + sourceName, 'success');
        }

        function showPredictions(geojson) {
            if (predLayer) map.removeLayer(predLayer);
            predLayer = L.geoJSON(geojson, {
                style: function(feature) {
                    const bound = feature.properties && feature.properties.bound;
                    return {
                        color: bound ? '#ff6600' : '#ff0000',
                        weight: 2,
                        opacity: 0.9,
                        fillColor: bound ? '#ff6600' : '#ff0000',
                        fillOpacity: bound ? 0.5 : 0.3
                    };
                },
                onEachFeature: function(feature, layer) {
                    if (feature.properties) {
                        let popup = '<b>Polygon #' + (feature.properties.id || '?') + '</b><br>';
                        if (feature.properties.bound) {
                            popup += '<span style="color:#0f0">✓ Bound to graph</span><br>';
                            if (feature.properties.edge_id !== undefined) popup += 'Edge ID: ' + feature.properties.edge_id + '<br>';
                            // Show road properties if available
                            const skip = ['id', 'bound', 'edge_id'];
                            for (const [k, v] of Object.entries(feature.properties)) {
                                if (!skip.includes(k) && v !== null && v !== undefined) {
                                    popup += k + ': ' + v + '<br>';
                                }
                            }
                        } else {
                            popup += '<span style="color:#f00">✗ Not bound</span>';
                        }
                        layer.bindPopup(popup);
                    }
                }
            }).addTo(map);
        }

        btnTrain.onclick = async () => {
            if (!currentBbox) return;
            const epochs = parseInt(document.getElementById('epochs').value) || 10;
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            setStatus('Starting training...', 'info');
            await fetch('/api/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({bbox: currentBbox, epochs: epochs})
            });
        };

        btnPredict.onclick = async () => {
            if (!currentBbox) return;
            progressContainer.style.display = 'block';
            progressFill.style.width = '0%';
            setStatus('Starting detection...', 'info');
            const bindToGraph = document.getElementById('bind-graph').checked;
            const source = document.getElementById('source-select').value;
            await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    bbox: currentBbox,
                    bind_to_graph: bindToGraph,
                    source: source
                })
            });
        };

        // Data source handling
        const sourceSelect = document.getElementById('source-select');
        const uploadInfo = document.getElementById('upload-info');
        const fileInput = document.getElementById('geojson-upload');

        document.getElementById('btn-upload').onclick = () => fileInput.click();

        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            setStatus('Uploading GeoJSON...', 'info');
            const formData = new FormData();
            formData.append('file', file);
            try {
                const resp = await fetch('/api/sources/upload', {method: 'POST', body: formData});
                const result = await resp.json();
                if (result.error) {
                    setStatus('Error: ' + result.error, 'error');
                } else {
                    setStatus('GeoJSON uploaded: ' + result.features_count + ' features', 'success');
                    uploadInfo.innerHTML = '<b>File:</b> ' + file.name + '<br><b>Features:</b> ' + result.features_count + '<br><b>Properties:</b> ' + result.properties.slice(0,5).join(', ');
                    uploadInfo.style.display = 'block';
                    // Enable and select custom option
                    sourceSelect.querySelector('option[value="custom"]').disabled = false;
                    sourceSelect.value = 'custom';
                    if (currentBbox) loadGroundTruth();
                }
            } catch(err) {
                setStatus('Upload failed: ' + err, 'error');
            }
        };

        sourceSelect.onchange = async () => {
            await fetch('/api/sources/' + sourceSelect.value, {method: 'POST'});
            if (currentBbox) {
                setStatus('Loading roads from ' + sourceSelect.options[sourceSelect.selectedIndex].text + '...', 'info');
                await loadGroundTruth();
            }
        };

        // Model management
        const modelsList = document.getElementById('models-list');

        async function loadModels() {
            try {
                const resp = await fetch('/api/models');
                const data = await resp.json();
                if (data.models.length === 0) {
                    modelsList.innerHTML = '<span style="color:#888">No models yet. Train one!</span>';
                    return;
                }
                modelsList.innerHTML = data.models.map(m => {
                    const date = new Date(m.created).toLocaleDateString('ru-RU', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
                    const isBest = m.name === 'best_model';
                    return '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #444">' +
                        '<span>' + (isBest ? '⭐ ' : '') + m.name + '<br><small style="color:#666">' + date + ' | ' + m.size_mb.toFixed(1) + 'MB</small></span>' +
                        '<div>' +
                            (isBest ? '' : '<button onclick="activateModel(\'' + m.name + '\')" style="background:#00d4ff;border:none;color:#000;padding:3px 6px;border-radius:3px;cursor:pointer;margin-right:3px;font-size:10px">Use</button>') +
                            '<button onclick="deleteModel(\'' + m.name + '\')" style="background:#ff4757;border:none;color:#fff;padding:3px 6px;border-radius:3px;cursor:pointer;font-size:10px">✕</button>' +
                        '</div>' +
                    '</div>';
                }).join('');
            } catch(err) {
                modelsList.innerHTML = '<span style="color:#ff4757">Error loading models</span>';
            }
        }

        async function deleteModel(name) {
            if (!confirm('Delete model "' + name + '"?')) return;
            await fetch('/api/models/' + name, {method: 'DELETE'});
            loadModels();
            setStatus('Model ' + name + ' deleted', 'info');
        }

        async function activateModel(name) {
            const resp = await fetch('/api/models/' + name + '/activate', {method: 'POST'});
            const result = await resp.json();
            if (result.error) {
                setStatus('Error: ' + result.error, 'error');
            } else {
                setStatus('Model ' + name + ' activated', 'success');
                loadModels();
            }
        }

        document.getElementById('btn-refresh-models').onclick = loadModels;

        document.getElementById('btn-delete-all').onclick = async () => {
            if (!confirm('Delete ALL models? This cannot be undone!')) return;
            const resp = await fetch('/api/models', {method: 'DELETE'});
            const result = await resp.json();
            setStatus('Deleted ' + result.count + ' models', 'info');
            loadModels();
        };

        // Load models on start
        loadModels();
    </script>
</body>
</html>"""


def run_app(host: str = "0.0.0.0", port: int = 8080):
    """Run the application."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_app()