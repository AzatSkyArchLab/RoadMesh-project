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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import box

# Create app
app = FastAPI(title="RoadMesh", version="0.1.0")


# Global state
class AppState:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.jobs: dict[str, dict] = {}
        self.models: dict[str, Path] = {}

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


@app.get("/api/ground-truth")
async def get_ground_truth(minx: float, miny: float, maxx: float, maxy: float):
    """Get ground truth roads for bbox."""
    vector_path = Path("data/geojson_data/osi_sush.geojson")
    
    if not vector_path.exists():
        return {"type": "FeatureCollection", "features": []}
    
    bbox = (minx, miny, maxx, maxy)
    gdf = gpd.read_file(vector_path, bbox=bbox)
    
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    
    bbox_geom = box(*bbox)
    gdf = gdf[gdf.intersects(bbox_geom)]
    
    return json.loads(gdf.to_json())


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
    """Run inference on area."""
    print(f"\n{'='*60}")
    print(f"[PREDICT] Starting job {job_id}")
    print(f"[PREDICT] BBox: {request.bbox}")
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
        from roadmesh.geometry.vectorizer import Vectorizer
        
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
        model.eval()
        
        await update(10, "Fetching satellite tiles...")
        
        bbox_config = BBoxConfig(minx=bbox.minx, miny=bbox.miny, maxx=bbox.maxx, maxy=bbox.maxy)
        fetcher = TileFetcher(provider="esri", cache_dir=Path("data/cache/tiles"))
        
        image, metadata = await fetcher.fetch_bbox_async(bbox_config, zoom=18)
        print(f"[PREDICT] Fetched image: {image.shape}")
        
        await update(40, "Running inference...")
        
        img_resized = cv2.resize(image, (512, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    output = model(img_tensor)
            else:
                output = model(img_tensor)
            
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (pred > 0.5).astype(np.uint8)
        
        print(f"[PREDICT] Mask shape: {mask.shape}, road pixels: {mask.sum()}")
        
        await update(70, "Vectorizing results...")
        
        vectorizer = Vectorizer()
        polygons_px = vectorizer.mask_to_polygons(mask * 255)
        
        print(f"[PREDICT] Found {len(polygons_px)} polygons")
        
        if polygons_px:
            polygons_geo = vectorizer.polygons_to_geo(
                polygons_px,
                bbox_config,
                (mask.shape[1], mask.shape[0])
            )
            geojson = vectorizer.to_geojson(polygons_geo)
        else:
            geojson = {"type": "FeatureCollection", "features": []}
        
        await update(90, "Saving results...")
        
        pred_dir = Path("data/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        with open(pred_dir / f"{job_id}.geojson", "w") as f:
            json.dump(geojson, f)
        
        state.jobs[job_id]["result"] = {
            "geojson": geojson,
            "polygon_count": len(geojson["features"])
        }
        
        await update(100, f"Done! Found {len(geojson['features'])} road polygons", "completed")
        
        await fetcher.close()
        
        print(f"[PREDICT] Completed! Found {len(geojson['features'])} polygons")
        
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
        <h3>2. TRAIN MODEL</h3>
        <label>Epochs:</label>
        <input type="number" id="epochs" value="10" min="1" max="200">
        <button class="btn btn-primary" id="btn-train" disabled>🧠 Train Model</button>
        <h3>3. DETECT ROADS</h3>
        <button class="btn btn-success" id="btn-predict" disabled>🔍 Detect Roads</button>
        <div class="progress-container" id="progress">
            <div class="progress-text" id="progress-message">Processing...</div>
            <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
            <div class="progress-text"><span id="progress-percent">0</span>%</div>
        </div>
        <div class="legend">
            <strong>Legend</strong>
            <div class="legend-item"><div class="legend-color" style="background:#00ff00"></div>Ground Truth</div>
            <div class="legend-item"><div class="legend-color" style="background:#ff0000"></div>Predictions</div>
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
            const url = '/api/ground-truth?minx=' + currentBbox.minx + '&miny=' + currentBbox.miny + '&maxx=' + currentBbox.maxx + '&maxy=' + currentBbox.maxy;
            const resp = await fetch(url);
            const geojson = await resp.json();
            if (gtLayer) map.removeLayer(gtLayer);
            gtLayer = L.geoJSON(geojson, {style: {color: '#00ff00', weight: 3, opacity: 0.8}}).addTo(map);
            setStatus('Loaded ' + geojson.features.length + ' road segments', 'success');
        }

        function showPredictions(geojson) {
            if (predLayer) map.removeLayer(predLayer);
            predLayer = L.geoJSON(geojson, {style: {color: '#ff0000', weight: 2, opacity: 0.9, fillColor: '#ff0000', fillOpacity: 0.4}}).addTo(map);
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
            await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({bbox: currentBbox})
            });
        };
    </script>
</body>
</html>"""


def run_app(host: str = "0.0.0.0", port: int = 8080):
    """Run the application."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_app()