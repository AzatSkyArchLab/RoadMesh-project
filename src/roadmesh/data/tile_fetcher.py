"""
Satellite Tile Fetcher

Downloads and caches satellite imagery tiles from various providers.
Primary support for Esri World Imagery (free).
"""
from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import httpx
import mercantile
import numpy as np
from diskcache import Cache
from PIL import Image
import io

from roadmesh.core.config import BBox, DataConfig


class TileProvider(ABC):
    """Abstract base class for tile providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def max_zoom(self) -> int:
        """Maximum supported zoom level."""
        pass
    
    @abstractmethod
    def get_tile_url(self, x: int, y: int, z: int) -> str:
        """Get URL for specific tile coordinates."""
        pass


class EsriWorldImagery(TileProvider):
    """
    Esri World Imagery - free for non-commercial use.
    High quality, good coverage, no API key required.
    """
    
    name = "esri"
    max_zoom = 19
    
    # Multiple server subdomains for parallel downloads
    BASE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    
    def get_tile_url(self, x: int, y: int, z: int) -> str:
        return self.BASE_URL.format(x=x, y=y, z=z)


class BingAerial(TileProvider):
    """Bing Maps Aerial imagery."""
    
    name = "bing"
    max_zoom = 19
    
    def get_tile_url(self, x: int, y: int, z: int) -> str:
        quadkey = self._tile_to_quadkey(x, y, z)
        subdomain = quadkey[-1] if quadkey else "0"
        return f"https://ecn.t{subdomain}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1"
    
    @staticmethod
    def _tile_to_quadkey(x: int, y: int, z: int) -> str:
        """Convert tile coordinates to Bing quadkey."""
        quadkey = []
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (x & mask) != 0:
                digit += 1
            if (y & mask) != 0:
                digit += 2
            quadkey.append(str(digit))
        return "".join(quadkey)


class GoogleSatellite(TileProvider):
    """Google Maps Satellite (requires API key for high volume)."""
    
    name = "google"
    max_zoom = 20
    
    def get_tile_url(self, x: int, y: int, z: int) -> str:
        return f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"


# Provider registry
PROVIDERS: dict[str, type[TileProvider]] = {
    "esri": EsriWorldImagery,
    "bing": BingAerial,
    "google": GoogleSatellite,
}


class TileFetcher:
    """
    Downloads and caches satellite imagery tiles.
    
    Features:
    - Disk caching with configurable size
    - Async parallel downloads
    - Multiple provider support
    - Bbox to tile conversion
    """
    
    def __init__(
        self,
        provider: str = "esri",
        cache_dir: Optional[Path] = None,
        cache_size_gb: float = 10.0,
    ):
        self.provider = PROVIDERS[provider]()
        self.cache_dir = cache_dir or Path("./data/cache/tiles")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize disk cache
        self._cache = Cache(
            str(self.cache_dir),
            size_limit=int(cache_size_gb * 1024 * 1024 * 1024)
        )
        
        # HTTP client settings
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=20),
                headers={
                    "User-Agent": "RoadMesh/1.0 (satellite imagery research)"
                }
            )
        return self._client
    
    def _get_cache_key(self, x: int, y: int, z: int) -> str:
        """Generate cache key for tile."""
        return f"{self.provider.name}_{z}_{x}_{y}"
    
    async def fetch_tile_async(
        self,
        x: int,
        y: int,
        z: int,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Fetch a single tile asynchronously.
        
        Args:
            x, y, z: Tile coordinates
            use_cache: Whether to use disk cache
            
        Returns:
            RGB numpy array (H, W, 3)
        """
        cache_key = self._get_cache_key(x, y, z)
        
        # Check cache first
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Download tile
        url = self.provider.get_tile_url(x, y, z)
        client = await self._get_client()
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            tile = np.array(image)
            
            # Cache result
            if use_cache:
                self._cache[cache_key] = tile
            
            return tile
            
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to fetch tile ({x}, {y}, {z}): {e}")
    
    def fetch_tile(self, x: int, y: int, z: int, use_cache: bool = True) -> np.ndarray:
        """Synchronous wrapper for fetch_tile_async."""
        return asyncio.run(self.fetch_tile_async(x, y, z, use_cache))
    
    async def fetch_tiles_async(
        self,
        tiles: list[tuple[int, int, int]],
        use_cache: bool = True,
        progress_callback: Optional[callable] = None
    ) -> list[np.ndarray]:
        """
        Fetch multiple tiles in parallel.
        
        Args:
            tiles: List of (x, y, z) tuples
            use_cache: Whether to use disk cache
            progress_callback: Optional callback(completed, total)
            
        Returns:
            List of RGB numpy arrays
        """
        async def fetch_with_progress(tile, idx):
            result = await self.fetch_tile_async(*tile, use_cache)
            if progress_callback:
                progress_callback(idx + 1, len(tiles))
            return result
        
        tasks = [fetch_with_progress(tile, i) for i, tile in enumerate(tiles)]
        return await asyncio.gather(*tasks)
    
    def bbox_to_tiles(self, bbox: BBox, zoom: int) -> list[tuple[int, int, int]]:
        """
        Convert bounding box to list of tile coordinates.
        
        Args:
            bbox: Geographic bounding box
            zoom: Zoom level
            
        Returns:
            List of (x, y, z) tile coordinates
        """
        tiles = list(mercantile.tiles(
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, zoom
        ))
        return [(t.x, t.y, t.z) for t in tiles]
    
    async def fetch_bbox_async(
        self,
        bbox: BBox,
        zoom: int,
        use_cache: bool = True
    ) -> tuple[np.ndarray, dict]:
        """
        Fetch all tiles covering a bounding box and stitch them.
        
        Args:
            bbox: Geographic bounding box
            zoom: Zoom level
            use_cache: Whether to use disk cache
            
        Returns:
            Tuple of (stitched image, metadata dict)
        """
        tiles = self.bbox_to_tiles(bbox, zoom)
        
        if not tiles:
            raise ValueError(f"No tiles found for bbox: {bbox}")
        
        # Get tile bounds
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Fetch all tiles
        tile_images = await self.fetch_tiles_async(tiles, use_cache)
        
        # Create tile lookup
        tile_dict = {(t[0], t[1]): img for t, img in zip(tiles, tile_images)}
        
        # Get tile size from first image
        tile_size = tile_images[0].shape[0]
        
        # Stitch tiles
        width = (max_x - min_x + 1) * tile_size
        height = (max_y - min_y + 1) * tile_size
        stitched = np.zeros((height, width, 3), dtype=np.uint8)
        
        for (x, y), img in tile_dict.items():
            px = (x - min_x) * tile_size
            py = (y - min_y) * tile_size
            stitched[py:py + tile_size, px:px + tile_size] = img
        
        # Calculate actual bounds of stitched image
        nw_tile = mercantile.bounds(min_x, min_y, zoom)
        se_tile = mercantile.bounds(max_x, max_y, zoom)
        
        metadata = {
            "zoom": zoom,
            "tiles": tiles,
            "bounds": {
                "west": nw_tile.west,
                "north": nw_tile.north,
                "east": se_tile.east,
                "south": se_tile.south,
            },
            "pixel_size": width,
            "tile_size": tile_size,
        }
        
        return stitched, metadata
    
    def fetch_bbox(
        self,
        bbox: BBox,
        zoom: int,
        use_cache: bool = True
    ) -> tuple[np.ndarray, dict]:
        """Synchronous wrapper for fetch_bbox_async."""
        return asyncio.run(self.fetch_bbox_async(bbox, zoom, use_cache))
    
    def tile_to_geo_bounds(self, x: int, y: int, z: int) -> BBox:
        """Get geographic bounds for a tile."""
        bounds = mercantile.bounds(x, y, z)
        return BBox(
            minx=bounds.west,
            miny=bounds.south,
            maxx=bounds.east,
            maxy=bounds.north
        )
    
    def clear_cache(self) -> None:
        """Clear all cached tiles."""
        self._cache.clear()
    
    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size_mb": self._cache.volume() / (1024 * 1024),
            "count": len(self._cache),
        }
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._client:
            try:
                asyncio.run(self.close())
            except RuntimeError:
                pass  # Event loop already closed


# Convenience function
def create_fetcher(config: Optional[DataConfig] = None) -> TileFetcher:
    """Create TileFetcher from config."""
    config = config or DataConfig()
    return TileFetcher(
        provider=config.tile_provider,
        cache_dir=config.cache_dir / "tiles"
    )
