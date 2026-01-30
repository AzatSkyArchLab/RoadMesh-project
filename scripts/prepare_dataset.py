#!/usr/bin/env python
"""
Dataset Preparation Script

Downloads satellite tiles and creates training dataset from vector road data.

Usage:
    python scripts/prepare_dataset.py \
        --vector-path data/raw/moscow_roads.geojson \
        --bbox 37.35,55.55,37.85,55.95 \
        --output-dir data/processed \
        --zoom 18
"""
from __future__ import annotations

import argparse
from pathlib import Path

from roadmesh.core.config import BBox, DataConfig
from roadmesh.data.tile_fetcher import TileFetcher
from roadmesh.data.label_processor import LabelProcessor, DatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare training dataset from vector road data"
    )
    
    parser.add_argument(
        "--vector-path",
        type=Path,
        required=True,
        help="Path to vector road data (GeoJSON, Shapefile, GeoPackage)"
    )
    
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box as 'minx,miny,maxx,maxy' (WGS84)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for dataset"
    )
    
    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="Zoom level for tiles (default: 18)"
    )
    
    parser.add_argument(
        "--tile-provider",
        type=str,
        default="esri",
        choices=["esri", "bing", "google"],
        help="Satellite tile provider"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory for tiles"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set fraction"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set fraction"
    )
    
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.01,
        help="Minimum road pixel coverage to include tile"
    )
    
    parser.add_argument(
        "--width-column",
        type=str,
        default=None,
        help="Column name for road width in vector data"
    )
    
    parser.add_argument(
        "--type-column", 
        type=str,
        default=None,
        help="Column name for road type in vector data"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse bbox
    bbox_parts = [float(x) for x in args.bbox.split(",")]
    if len(bbox_parts) != 4:
        raise ValueError("Bbox must be 'minx,miny,maxx,maxy'")
    
    bbox = BBox(
        minx=bbox_parts[0],
        miny=bbox_parts[1],
        maxx=bbox_parts[2],
        maxy=bbox_parts[3],
    )
    
    print(f"\n{'='*60}")
    print("RoadMesh Dataset Preparation")
    print(f"{'='*60}")
    print(f"Vector data: {args.vector_path}")
    print(f"Bounding box: {bbox.tuple}")
    print(f"Zoom level: {args.zoom}")
    print(f"Provider: {args.tile_provider}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Initialize components
    print("Initializing tile fetcher...")
    tile_fetcher = TileFetcher(
        provider=args.tile_provider,
        cache_dir=args.cache_dir / "tiles",
    )
    
    print("Initializing label processor...")
    label_processor = LabelProcessor()
    
    # Create dataset builder
    builder = DatasetBuilder(
        tile_fetcher=tile_fetcher,
        label_processor=label_processor,
        output_dir=args.output_dir,
    )
    
    # Build dataset
    print("\nBuilding dataset...")
    counts = builder.build_dataset(
        vector_path=args.vector_path,
        bbox=bbox,
        zoom=args.zoom,
        train_split=args.train_split,
        val_split=args.val_split,
        skip_empty=True,
        min_road_coverage=args.min_coverage,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Training samples:   {counts['train']}")
    print(f"Validation samples: {counts['val']}")
    print(f"Test samples:       {counts['test']}")
    print(f"Total:              {sum(counts.values())}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
