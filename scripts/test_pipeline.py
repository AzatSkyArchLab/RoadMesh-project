#!/usr/bin/env python
"""
Quick Test Script

Tests the full pipeline on a small area (1-2 km²) to verify everything works.
Respects Esri rate limits with delays between requests.

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --bbox 37.61,55.74,37.63,55.76
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np


# Test areas in Moscow (small, ~1-2 km²)
TEST_AREAS = {
    "kremlin": {
        "name": "Кремль и окрестности",
        "bbox": (37.609, 55.748, 37.625, 55.758),
        "description": "Центр Москвы, хорошее покрытие дорог"
    },
    "tverskaya": {
        "name": "Тверская улица", 
        "bbox": (37.600, 55.760, 37.615, 55.770),
        "description": "Плотная дорожная сеть"
    },
    "vdnkh": {
        "name": "ВДНХ",
        "bbox": (37.620, 55.825, 37.640, 55.840),
        "description": "Парковая зона + дороги"
    },
    "msu": {
        "name": "МГУ",
        "bbox": (37.525, 55.695, 37.545, 55.710),
        "description": "Воробьёвы горы"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test RoadMesh pipeline")
    
    parser.add_argument(
        "--area",
        type=str,
        default="kremlin",
        choices=list(TEST_AREAS.keys()),
        help="Predefined test area"
    )
    
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Custom bbox: 'minx,miny,maxx,maxy'"
    )
    
    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="Zoom level (default: 18)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test_output"),
        help="Output directory"
    )
    
    parser.add_argument(
        "--skip-tiles",
        action="store_true",
        help="Skip tile download (use cached)"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between tile requests (seconds)"
    )
    
    return parser.parse_args()


async def test_tile_fetcher(bbox, zoom: int, output_dir: Path, rate_limit: float):
    """Test tile fetching with rate limiting."""
    from roadmesh.core.config import BBox
    from roadmesh.data.tile_fetcher import TileFetcher
    
    print("\n" + "="*60)
    print("1. Testing Tile Fetcher")
    print("="*60)
    
    bbox_obj = BBox(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
    
    fetcher = TileFetcher(
        provider="esri",
        cache_dir=output_dir / "cache"
    )
    
    # Get tile list
    tiles = fetcher.bbox_to_tiles(bbox_obj, zoom)
    print(f"Tiles to fetch: {len(tiles)}")
    print(f"Estimated time: {len(tiles) * rate_limit:.1f}s (with rate limiting)")
    
    # Fetch with rate limiting
    images = []
    for i, (x, y, z) in enumerate(tiles):
        print(f"  Fetching tile {i+1}/{len(tiles)}: ({x}, {y}, {z})...", end=" ")
        
        try:
            img = await fetcher.fetch_tile_async(x, y, z, use_cache=True)
            images.append(img)
            print(f"✓ {img.shape}")
        except Exception as e:
            print(f"✗ {e}")
            images.append(None)
        
        # Rate limiting (skip if cached)
        if i < len(tiles) - 1:
            await asyncio.sleep(rate_limit)
    
    # Stitch tiles
    print("\nStitching tiles...")
    stitched, metadata = await fetcher.fetch_bbox_async(bbox_obj, zoom)
    
    # Save stitched image
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    img = Image.fromarray(stitched)
    img_path = output_dir / "stitched_satellite.png"
    img.save(img_path)
    print(f"Saved: {img_path} ({stitched.shape})")
    
    await fetcher.close()
    
    return stitched, metadata


def test_vectorizer(mask: np.ndarray, bbox, output_dir: Path):
    """Test mask vectorization."""
    from roadmesh.core.config import BBox
    from roadmesh.geometry.vectorizer import Vectorizer, MeshBuilder
    
    print("\n" + "="*60)
    print("3. Testing Vectorizer")
    print("="*60)
    
    bbox_obj = BBox(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
    
    vectorizer = Vectorizer()
    mesh_builder = MeshBuilder()
    
    # Extract polygons
    print("Extracting polygons from mask...")
    polygons_px = vectorizer.mask_to_polygons(mask)
    print(f"  Found {len(polygons_px)} polygons in pixel coords")
    
    # Transform to geo coords
    pixel_size = (mask.shape[1], mask.shape[0])
    polygons_geo = vectorizer.polygons_to_geo(polygons_px, bbox_obj, pixel_size)
    print(f"  Transformed to geo coords: {len(polygons_geo)} polygons")
    
    # Export GeoJSON
    geojson = vectorizer.to_geojson(polygons_geo)
    
    import json
    geojson_path = output_dir / "roads.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  Saved: {geojson_path}")
    
    # Export mesh
    mesh = mesh_builder.polygons_to_mesh_collection(polygons_geo)
    
    mesh_path = output_dir / "roads_mesh.json"
    with open(mesh_path, "w") as f:
        json.dump(mesh, f)
    print(f"  Saved: {mesh_path}")
    
    return geojson, mesh


def test_model_inference(image: np.ndarray, output_dir: Path):
    """Test model inference (or create dummy mask if no model)."""
    import torch
    
    print("\n" + "="*60)
    print("2. Testing Model Inference")
    print("="*60)
    
    # Check for trained model
    model_path = Path("checkpoints/best_model.pt")
    
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        from roadmesh.models.architectures import create_model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model(
            architecture="dlinknet34",
            checkpoint_path=str(model_path),
            device=device
        )
        model.eval()
        
        # Preprocess
        import cv2
        img_resized = cv2.resize(image, (512, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (pred > 0.5).astype(np.uint8) * 255
        
        print(f"  Inference complete: {mask.shape}")
        print(f"  Road pixels: {(mask > 0).sum()} ({(mask > 0).mean()*100:.1f}%)")
        
    else:
        print("No trained model found. Creating synthetic mask for testing...")
        print("(Train a model first with: python scripts/train.py)")
        
        # Create synthetic road mask for testing vectorization
        import cv2
        h, w = 512, 512
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw some synthetic roads
        cv2.line(mask, (50, 256), (462, 256), 255, 20)   # Horizontal
        cv2.line(mask, (256, 50), (256, 462), 255, 15)   # Vertical
        cv2.line(mask, (100, 100), (400, 400), 255, 12)  # Diagonal
        cv2.line(mask, (100, 400), (400, 100), 255, 12)  # Diagonal
        
        # Add some curves
        for i in range(50, 450, 5):
            y = int(150 + 50 * np.sin(i / 30))
            cv2.circle(mask, (i, y), 6, 255, -1)
        
        print(f"  Created synthetic mask: {mask.shape}")
    
    # Save mask
    from PIL import Image
    mask_path = output_dir / "road_mask.png"
    Image.fromarray(mask).save(mask_path)
    print(f"  Saved: {mask_path}")
    
    return mask


def test_dataset_creation(bbox, output_dir: Path):
    """Test dataset creation with synthetic labels."""
    print("\n" + "="*60)
    print("4. Testing Dataset Structure")
    print("="*60)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_dir / "dataset" / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset" / split / "masks").mkdir(parents=True, exist_ok=True)
    
    print("  Created dataset directory structure:")
    print(f"    {output_dir}/dataset/train/images/")
    print(f"    {output_dir}/dataset/train/masks/")
    print(f"    {output_dir}/dataset/val/...")
    print(f"    {output_dir}/dataset/test/...")
    
    print("\n  To create a real dataset, run:")
    print(f"    python scripts/prepare_dataset.py \\")
    print(f"      --vector-path <your_roads.geojson> \\")
    print(f"      --bbox {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} \\")
    print(f"      --output-dir {output_dir}/dataset")


def print_summary(output_dir: Path, test_area: dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"\nTest area: {test_area['name']}")
    print(f"Description: {test_area['description']}")
    print(f"Bbox: {test_area['bbox']}")
    
    print(f"\nOutput files in {output_dir}/:")
    
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  • {f.name} ({size_kb:.1f} KB)")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Подготовь свои векторные данные дорог (GeoJSON/Shapefile)
   - Должны содержать геометрию дорог для области Москвы
   - Положи в data/raw/moscow_roads.geojson

2. Создай датасет:
   python scripts/prepare_dataset.py \\
     --vector-path data/raw/moscow_roads.geojson \\
     --bbox 37.35,55.55,37.85,55.95 \\
     --output-dir data/processed

3. Обучи модель:
   python scripts/train.py \\
     --config configs/model/dlinknet_8gb.yaml

4. Запусти API:
   python -m roadmesh.api.app
""")


async def main():
    args = parse_args()
    
    # Get bbox
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        test_area = {"name": "Custom", "bbox": bbox, "description": "User-defined area"}
    else:
        test_area = TEST_AREAS[args.area]
        bbox = test_area["bbox"]
    
    print("\n" + "="*60)
    print("ROADMESH PIPELINE TEST")
    print("="*60)
    print(f"Area: {test_area['name']}")
    print(f"Bbox: {bbox}")
    print(f"Zoom: {args.zoom}")
    print(f"Output: {args.output_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Tile fetching
    if not args.skip_tiles:
        image, metadata = await test_tile_fetcher(
            bbox, args.zoom, args.output_dir, args.rate_limit
        )
    else:
        print("\nSkipping tile download (--skip-tiles)")
        # Load cached
        from PIL import Image
        img_path = args.output_dir / "stitched_satellite.png"
        if img_path.exists():
            image = np.array(Image.open(img_path))
        else:
            print("No cached image found. Run without --skip-tiles first.")
            return
    
    # Test 2: Model inference (or synthetic mask)
    mask = test_model_inference(image, args.output_dir)
    
    # Test 3: Vectorization
    test_vectorizer(mask, bbox, args.output_dir)
    
    # Test 4: Dataset structure
    test_dataset_creation(bbox, args.output_dir)
    
    # Summary
    print_summary(args.output_dir, test_area)


if __name__ == "__main__":
    asyncio.run(main())
