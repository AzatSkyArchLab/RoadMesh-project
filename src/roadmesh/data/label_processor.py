"""
Label Processor

Converts vector road data (GeoJSON, Shapefile, GeoPackage) to raster masks
for training the segmentation model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping
from shapely.ops import unary_union
import cv2

from roadmesh.core.config import BBox, GeometryConfig


class LabelProcessor:
    """
    Processes vector road data into raster masks for model training.
    
    Supports:
    - GeoJSON, Shapefile, GeoPackage input
    - Road width estimation from attributes
    - Distance transform generation
    - Orientation field generation (for connectivity)
    """
    
    def __init__(
        self,
        config: Optional[GeometryConfig] = None,
        default_road_width: float = 6.0,  # meters
    ):
        self.config = config or GeometryConfig()
        self.default_road_width = default_road_width
        
        # Ширина дорог по СП 42.13330.2016 (метры)
        # Сопоставление с классификацией kl_gp Москвы
        self.road_widths = {
            # Магистральные общегородские (4-6 полос × 3.5-3.75 м)
            "Магистральные улицы общегородского значения I класса": 22.5,           # 6 × 3.75
            "Магистральные улицы общегородского значения II класса": 14.0,          # 4 × 3.5
            "Магистральные улицы общегородского значения непрерывного движения": 22.5,  # 6 × 3.75
            "Магистральные улицы общегородского значения регулируемого движения": 14.0, # 4 × 3.5
            # Центр (обычно меньше полос)
            "Магистральные улицы общегородского значения I класса центра": 15.0,    # 4 × 3.75
            "Магистральные улицы общегородского значения II класса центра": 10.5,   # 3 × 3.5
            # Районные (2-4 полосы × 3.5 м)
            "Магистральные улицы районного значения": 10.5,                         # 3 × 3.5
            "Магистральные улицы районного значения центра": 7.0,                   # 2 × 3.5
            # Местные (2 полосы × 3.0 м)
            "Прочая улично-дорожная сеть Москвы": 6.0,                              # 2 × 3.0
            "без категории": 6.0,                                                   # 2 × 3.0
            "default": 6.0,
        }
    
    def load_vectors(
        self,
        path: Union[str, Path],
        bbox: Optional[BBox] = None
    ) -> gpd.GeoDataFrame:
        """
        Load vector data from file, optionally clipped to bbox.
        
        Args:
            path: Path to vector file (GeoJSON, SHP, GPKG)
            bbox: Optional bounding box to clip data
            
        Returns:
            GeoDataFrame with road geometries
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Vector file not found: {path}")
        
        # Read with optional bbox filter
        if bbox:
            gdf = gpd.read_file(
                path,
                bbox=(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
            )
        else:
            gdf = gpd.read_file(path)
        
        # Ensure WGS84 CRS
        if gdf.crs is None:
            gdf.set_crs(self.config.default_crs, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(self.config.default_crs)
        
        return gdf
    
    def estimate_road_width(
        self,
        gdf: gpd.GeoDataFrame,
        width_column: Optional[str] = None,
        type_column: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Estimate road width for each feature.
        """
        gdf = gdf.copy()
        
        # Handle empty GeoDataFrame
        if gdf.empty:
            gdf["width"] = []
            return gdf

        if width_column and width_column in gdf.columns:
            gdf["width"] = gdf[width_column].fillna(self.default_road_width)
        elif type_column and type_column in gdf.columns:
            # Don't use .lower() - Russian text is case-sensitive
            gdf["width"] = gdf[type_column].map(
                lambda x: self.road_widths.get(str(x), self.default_road_width)
            )
        else:
            gdf["width"] = self.default_road_width

        return gdf
    
    def buffer_roads(
        self,
        gdf: gpd.GeoDataFrame,
        use_width: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Buffer road centerlines to polygons based on width.
        
        Args:
            gdf: GeoDataFrame with road geometries and 'width' column
            use_width: Whether to use per-feature width or fixed buffer
            
        Returns:
            GeoDataFrame with polygon geometries
        """
        # Convert to metric CRS for buffering
        gdf_metric = gdf.to_crs(self.config.working_crs)
        
        if use_width and "width" in gdf_metric.columns:
            # Buffer each road by half its width
            gdf_metric["geometry"] = gdf_metric.apply(
                lambda row: row.geometry.buffer(row["width"] / 2, cap_style=2),
                axis=1
            )
        else:
            # Fixed buffer
            gdf_metric["geometry"] = gdf_metric.geometry.buffer(
                self.default_road_width / 2, cap_style=2
            )
        
        # Convert back to WGS84
        return gdf_metric.to_crs(self.config.default_crs)
    
    def rasterize(
        self,
        gdf: gpd.GeoDataFrame,
        bounds: BBox,
        shape: tuple[int, int],
        all_touched: bool = True
    ) -> np.ndarray:
        """
        Rasterize vector geometries to binary mask.
        
        Args:
            gdf: GeoDataFrame with geometries to rasterize
            bounds: Geographic bounds for the output raster
            shape: Output shape (height, width)
            all_touched: Whether to include all touched pixels
            
        Returns:
            Binary mask array (H, W) with dtype uint8
        """
        if gdf.empty:
            return np.zeros(shape, dtype=np.uint8)
        
        # Create transform
        transform = from_bounds(
            bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
            shape[1], shape[0]
        )
        
        # Rasterize all geometries
        shapes = [(mapping(geom), 1) for geom in gdf.geometry if geom is not None]
        
        if not shapes:
            return np.zeros(shape, dtype=np.uint8)
        
        mask = features.rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype=np.uint8
        )
        
        return mask
    
    def create_distance_transform(
        self,
        mask: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Create distance transform from binary mask.
        Useful for weighted loss or post-processing.
        
        Args:
            mask: Binary mask (H, W)
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Distance transform array (H, W)
        """
        # Invert mask (distance from background)
        inverted = 1 - mask
        
        # Calculate distance transform
        dist = cv2.distanceTransform(
            inverted.astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        )
        
        if normalize and dist.max() > 0:
            dist = dist / dist.max()
        
        return dist.astype(np.float32)
    
    def create_boundary_mask(
        self,
        mask: np.ndarray,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Create road boundary mask for boundary-aware loss.
        
        Args:
            mask: Binary road mask (H, W)
            thickness: Boundary thickness in pixels
            
        Returns:
            Boundary mask array (H, W)
        """
        # Dilate and erode
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(mask, kernel, iterations=thickness)
        eroded = cv2.erode(mask, kernel, iterations=thickness)
        
        # Boundary is difference
        boundary = dilated - eroded
        
        return boundary.astype(np.uint8)
    
    def create_orientation_field(
        self,
        mask: np.ndarray,
        gdf: Optional[gpd.GeoDataFrame] = None
    ) -> np.ndarray:
        """
        Create orientation field for connectivity-aware training.
        Each pixel encodes the local road direction as (cos θ, sin θ).
        
        Args:
            mask: Binary road mask (H, W)
            gdf: Optional GeoDataFrame for more accurate orientation
            
        Returns:
            Orientation field (H, W, 2)
        """
        h, w = mask.shape
        orientation = np.zeros((h, w, 2), dtype=np.float32)
        
        # Use Sobel gradients to estimate local orientation
        grad_x = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
        
        # Orientation perpendicular to gradient (along road)
        magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
        
        # Road direction is perpendicular to edge gradient
        orientation[:, :, 0] = -grad_y / magnitude  # cos
        orientation[:, :, 1] = grad_x / magnitude   # sin
        
        # Zero out non-road pixels
        orientation[mask == 0] = 0
        
        return orientation
    
    def process_tile(
        self,
        gdf: gpd.GeoDataFrame,
        bounds: BBox,
        tile_size: int = 512,
        include_distance: bool = False,
        include_boundary: bool = False,
        include_orientation: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Process vector data for a single tile.
        
        Args:
            gdf: GeoDataFrame with road geometries
            bounds: Tile geographic bounds
            tile_size: Output tile size in pixels
            include_distance: Include distance transform
            include_boundary: Include boundary mask
            include_orientation: Include orientation field
            
        Returns:
            Dictionary with mask and optional auxiliary outputs
        """
        shape = (tile_size, tile_size)
        
        # Clip to tile bounds
        tile_box = box(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy)
        gdf_clipped = gdf[gdf.intersects(tile_box)].copy()
        
        if not gdf_clipped.empty:
            gdf_clipped["geometry"] = gdf_clipped.intersection(tile_box)
        
        # Estimate width and buffer
        gdf_clipped = self.estimate_road_width(gdf_clipped)
        gdf_buffered = self.buffer_roads(gdf_clipped)
        
        # Create main mask
        mask = self.rasterize(gdf_buffered, bounds, shape)
        
        result = {"mask": mask}
        
        # Optional outputs
        if include_distance:
            result["distance"] = self.create_distance_transform(mask)
        
        if include_boundary:
            result["boundary"] = self.create_boundary_mask(mask)
        
        if include_orientation:
            result["orientation"] = self.create_orientation_field(mask, gdf_clipped)
        
        return result


class DatasetBuilder:
    """
    Builds training dataset from satellite tiles and vector labels.
    """
    
    def __init__(
        self,
        tile_fetcher,
        label_processor: LabelProcessor,
        output_dir: Path,
    ):
        from roadmesh.data.tile_fetcher import TileFetcher
        
        self.tile_fetcher: TileFetcher = tile_fetcher
        self.label_processor = label_processor
        self.output_dir = Path(output_dir)
        
        # Create output directories
        for split in ["train", "val", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "masks").mkdir(parents=True, exist_ok=True)
    
    def build_dataset(
        self,
        vector_path: Union[str, Path],
        bbox: BBox,
        zoom: int = 18,
        train_split: float = 0.8,
        val_split: float = 0.1,
        skip_empty: bool = True,
        min_road_coverage: float = 0.01
    ) -> dict[str, int]:
        """
        Build complete dataset from vector data and satellite imagery.
        
        Args:
            vector_path: Path to vector road data
            bbox: Area to process
            zoom: Zoom level for tiles
            train_split: Fraction for training
            val_split: Fraction for validation
            skip_empty: Skip tiles with no roads
            min_road_coverage: Minimum road pixel fraction to include tile
            
        Returns:
            Dictionary with counts per split
        """
        import random
        from tqdm import tqdm
        
        # Load vector data
        gdf = self.label_processor.load_vectors(vector_path, bbox)
        
        if gdf.empty:
            raise ValueError(f"No road data found in {vector_path} for bbox {bbox}")
        
        # Get all tiles
        tiles = self.tile_fetcher.bbox_to_tiles(bbox, zoom)
        
        # Shuffle for random split
        random.shuffle(tiles)
        
        # Calculate split indices
        n_train = int(len(tiles) * train_split)
        n_val = int(len(tiles) * val_split)
        
        splits = {
            "train": tiles[:n_train],
            "val": tiles[n_train:n_train + n_val],
            "test": tiles[n_train + n_val:]
        }
        
        counts = {"train": 0, "val": 0, "test": 0}
        
        for split_name, split_tiles in splits.items():
            for x, y, z in tqdm(split_tiles, desc=f"Processing {split_name}"):
                try:
                    # Get tile bounds
                    tile_bounds = self.tile_fetcher.tile_to_geo_bounds(x, y, z)
                    
                    # Process label
                    labels = self.label_processor.process_tile(gdf, tile_bounds)
                    mask = labels["mask"]
                    
                    # Check coverage
                    coverage = mask.sum() / mask.size
                    if skip_empty and coverage < min_road_coverage:
                        continue
                    
                    # Fetch satellite tile
                    image = self.tile_fetcher.fetch_tile(x, y, z)
                    
                    # Resize if needed
                    if image.shape[0] != 512:
                        image = cv2.resize(image, (512, 512))
                        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                    
                    # Save
                    tile_id = f"{z}_{x}_{y}"
                    
                    image_path = self.output_dir / split_name / "images" / f"{tile_id}.png"
                    mask_path = self.output_dir / split_name / "masks" / f"{tile_id}.png"
                    
                    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(mask_path), mask * 255)
                    
                    counts[split_name] += 1
                    
                except Exception as e:
                    print(f"Error processing tile ({x}, {y}, {z}): {e}")
                    continue
        
        return counts
