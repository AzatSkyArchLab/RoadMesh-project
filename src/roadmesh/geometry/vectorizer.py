"""
Vectorization Module

Converts segmentation masks to vector polygons and mesh format
compatible with Three.js.
"""
from __future__ import annotations

from typing import Optional, Union
import json

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
import geopandas as gpd

from roadmesh.core.config import GeometryConfig, BBox


class Vectorizer:
    """
    Converts binary segmentation masks to vector polygons.
    
    Pipeline:
    1. Morphological cleanup
    2. Contour extraction
    3. Polygon simplification
    4. Topology repair
    """
    
    def __init__(self, config: Optional[GeometryConfig] = None):
        self.config = config or GeometryConfig()
    
    def cleanup_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological operations to clean up mask.
        
        Args:
            mask: Binary mask (H, W)
            kernel_size: Morphological kernel size
            iterations: Number of iterations
            
        Returns:
            Cleaned binary mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        # Close small holes
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations
        )
        
        # Open to remove noise
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations
        )
        
        return mask
    
    def extract_contours(
        self,
        mask: np.ndarray
    ) -> list[np.ndarray]:
        """
        Extract contours from binary mask.
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            List of contour arrays
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1
        )
        
        return contours
    
    def contour_to_polygon(
        self,
        contour: np.ndarray,
        simplify: bool = True
    ) -> Optional[Polygon]:
        """
        Convert OpenCV contour to Shapely Polygon.
        
        Args:
            contour: OpenCV contour array
            simplify: Whether to apply Douglas-Peucker simplification
            
        Returns:
            Shapely Polygon or None if invalid
        """
        # Need at least 3 points for a polygon
        if len(contour) < 3:
            return None
        
        # Convert to coordinate list
        coords = contour.squeeze()
        if coords.ndim == 1:
            return None
        
        # Create polygon
        try:
            polygon = Polygon(coords)
            
            # Make valid if needed
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            
            # Filter by area
            if polygon.area < self.config.min_polygon_area:
                return None
            
            # Simplify if requested
            if simplify:
                polygon = polygon.simplify(
                    self.config.simplify_tolerance,
                    preserve_topology=True
                )
            
            return polygon
            
        except Exception:
            return None
    
    def mask_to_polygons(
        self,
        mask: np.ndarray,
        cleanup: bool = True,
        simplify: bool = True
    ) -> list[Polygon]:
        """
        Convert binary mask to list of polygons.
        
        Args:
            mask: Binary mask (H, W), values 0 or 1/255
            cleanup: Apply morphological cleanup
            simplify: Simplify resulting polygons
            
        Returns:
            List of Shapely Polygons
        """
        # Normalize mask to 0-255
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        
        # Threshold
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Cleanup
        if cleanup:
            mask = self.cleanup_mask(mask)
        
        # Extract contours
        contours = self.extract_contours(mask)
        
        # Convert to polygons
        polygons = []
        for contour in contours:
            polygon = self.contour_to_polygon(contour, simplify=simplify)
            if polygon is not None:
                polygons.append(polygon)
        
        return polygons
    
    def polygons_to_geo(
        self,
        polygons: list[Polygon],
        bounds: BBox,
        pixel_size: tuple[int, int]
    ) -> list[Polygon]:
        """
        Transform polygons from pixel coordinates to geographic coordinates.
        
        Args:
            polygons: List of polygons in pixel coordinates
            bounds: Geographic bounds of the image
            pixel_size: Image size (width, height) in pixels
            
        Returns:
            List of polygons in geographic coordinates
        """
        width, height = pixel_size
        
        # Calculate scale factors
        x_scale = (bounds.maxx - bounds.minx) / width
        y_scale = (bounds.maxy - bounds.miny) / height
        
        geo_polygons = []
        
        for polygon in polygons:
            # Transform coordinates
            coords = np.array(polygon.exterior.coords)
            
            # Pixel to geo: x -> lon, y -> lat (note: y is flipped)
            geo_coords = np.column_stack([
                bounds.minx + coords[:, 0] * x_scale,
                bounds.maxy - coords[:, 1] * y_scale  # Y flipped
            ])
            
            try:
                geo_polygon = Polygon(geo_coords)
                if geo_polygon.is_valid:
                    geo_polygons.append(geo_polygon)
                else:
                    geo_polygons.append(make_valid(geo_polygon))
            except Exception:
                continue
        
        return geo_polygons
    
    def to_geojson(
        self,
        polygons: list[Polygon],
        properties: Optional[list[dict]] = None
    ) -> dict:
        """
        Convert polygons to GeoJSON FeatureCollection.
        
        Args:
            polygons: List of Shapely polygons
            properties: Optional list of property dicts for each polygon
            
        Returns:
            GeoJSON FeatureCollection dict
        """
        features = []
        
        for i, polygon in enumerate(polygons):
            props = properties[i] if properties else {}
            
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "id": i,
                    **props
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


class MeshBuilder:
    """
    Converts 2D polygons to 3D mesh format for Three.js.
    """
    
    def __init__(self, default_elevation: float = 0.0):
        self.default_elevation = default_elevation
    
    def triangulate_polygon(
        self,
        polygon: Polygon
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangulate a polygon using ear clipping.
        
        Args:
            polygon: Shapely Polygon
            
        Returns:
            Tuple of (vertices, indices) arrays
        """
        from shapely import get_coordinates
        
        # Get exterior coordinates (remove closing point)
        coords = np.array(polygon.exterior.coords)[:-1]
        n_vertices = len(coords)
        
        if n_vertices < 3:
            return np.array([]), np.array([])
        
        # Simple ear clipping triangulation
        vertices = coords.copy()
        indices = []
        
        # For convex polygons, fan triangulation
        if polygon.is_valid and self._is_convex(coords):
            for i in range(1, n_vertices - 1):
                indices.extend([0, i, i + 1])
        else:
            # Fall back to fan triangulation from centroid
            # This works for most road shapes
            centroid = np.array(polygon.centroid.coords[0])
            
            # Add centroid as new vertex
            vertices = np.vstack([vertices, centroid])
            center_idx = n_vertices
            
            for i in range(n_vertices):
                next_i = (i + 1) % n_vertices
                indices.extend([center_idx, i, next_i])
        
        return vertices, np.array(indices, dtype=np.int32)
    
    def _is_convex(self, coords: np.ndarray) -> bool:
        """Check if polygon coordinates form a convex shape."""
        n = len(coords)
        if n < 3:
            return False
        
        sign = None
        for i in range(n):
            p1 = coords[i]
            p2 = coords[(i + 1) % n]
            p3 = coords[(i + 2) % n]
            
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            
            if abs(cross) > 1e-10:
                if sign is None:
                    sign = cross > 0
                elif (cross > 0) != sign:
                    return False
        
        return True
    
    def polygon_to_mesh(
        self,
        polygon: Polygon,
        elevation: Optional[float] = None,
        properties: Optional[dict] = None
    ) -> dict:
        """
        Convert polygon to Three.js BufferGeometry format.
        
        Args:
            polygon: Shapely Polygon
            elevation: Z coordinate for all vertices
            properties: Optional properties to include
            
        Returns:
            Dictionary with positions, indices, normals
        """
        elevation = elevation if elevation is not None else self.default_elevation
        
        # Triangulate
        vertices_2d, indices = self.triangulate_polygon(polygon)
        
        if len(vertices_2d) == 0:
            return None
        
        # Add Z coordinate
        n_vertices = len(vertices_2d)
        positions = np.zeros((n_vertices, 3), dtype=np.float32)
        positions[:, :2] = vertices_2d
        positions[:, 2] = elevation
        
        # Flat normals pointing up
        normals = np.zeros((n_vertices, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        
        # UVs based on bounding box
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        uvs = np.zeros((n_vertices, 2), dtype=np.float32)
        if width > 0 and height > 0:
            uvs[:, 0] = (vertices_2d[:, 0] - bounds[0]) / width
            uvs[:, 1] = (vertices_2d[:, 1] - bounds[1]) / height
        
        result = {
            "positions": positions.flatten().tolist(),
            "indices": indices.tolist(),
            "normals": normals.flatten().tolist(),
            "uvs": uvs.flatten().tolist(),
        }
        
        if properties:
            result["properties"] = properties
        
        return result
    
    def polygons_to_mesh_collection(
        self,
        polygons: list[Polygon],
        properties_list: Optional[list[dict]] = None,
        elevation: float = 0.0
    ) -> dict:
        """
        Convert multiple polygons to mesh collection.
        
        Args:
            polygons: List of Shapely polygons
            properties_list: Optional properties for each polygon
            elevation: Default elevation
            
        Returns:
            Dictionary with features array
        """
        features = []
        
        for i, polygon in enumerate(polygons):
            props = properties_list[i] if properties_list else {"id": i}
            
            mesh = self.polygon_to_mesh(polygon, elevation, props)
            
            if mesh is not None:
                features.append(mesh)
        
        return {
            "type": "MeshCollection",
            "features": features,
            "count": len(features)
        }
    
    def to_threejs_json(
        self,
        polygons: list[Polygon],
        properties_list: Optional[list[dict]] = None,
    ) -> str:
        """
        Export to Three.js compatible JSON string.
        """
        collection = self.polygons_to_mesh_collection(polygons, properties_list)
        return json.dumps(collection)


class GraphBinder:
    """
    Binds polygon meshes to road graph edges.
    """
    
    def __init__(
        self,
        graph: Optional[gpd.GeoDataFrame] = None,
        snap_tolerance: float = 5.0
    ):
        """
        Args:
            graph: GeoDataFrame with road edges (LineString geometries)
            snap_tolerance: Maximum distance to snap polygon to edge
        """
        self.graph = graph
        self.snap_tolerance = snap_tolerance
    
    def load_graph(self, path: str) -> None:
        """Load road graph from file."""
        self.graph = gpd.read_file(path)
    
    def find_nearest_edge(
        self,
        polygon: Polygon
    ) -> Optional[tuple[int, dict]]:
        """
        Find nearest edge in graph to polygon centroid.
        
        Returns:
            Tuple of (edge_index, edge_properties) or None
        """
        if self.graph is None or self.graph.empty:
            return None
        
        centroid = polygon.centroid
        
        # Calculate distances to all edges
        distances = self.graph.geometry.distance(centroid)
        
        # Find nearest
        min_idx = distances.idxmin()
        min_dist = distances[min_idx]
        
        if min_dist > self.snap_tolerance:
            return None
        
        # Get edge properties
        edge = self.graph.loc[min_idx]
        properties = edge.drop('geometry').to_dict()
        
        return min_idx, properties
    
    def bind_polygons(
        self,
        polygons: list[Polygon]
    ) -> list[dict]:
        """
        Bind list of polygons to graph edges.
        
        Returns:
            List of property dicts with edge bindings
        """
        bindings = []
        
        for polygon in polygons:
            result = self.find_nearest_edge(polygon)
            
            if result is not None:
                edge_idx, edge_props = result
                bindings.append({
                    "bound": True,
                    "edge_id": int(edge_idx),
                    **edge_props
                })
            else:
                bindings.append({
                    "bound": False,
                    "edge_id": None
                })
        
        return bindings


# Convenience function for full pipeline
def mask_to_mesh(
    mask: np.ndarray,
    bounds: BBox,
    graph: Optional[gpd.GeoDataFrame] = None,
    config: Optional[GeometryConfig] = None
) -> dict:
    """
    Full pipeline: mask → polygons → mesh with graph binding.
    
    Args:
        mask: Binary segmentation mask (H, W)
        bounds: Geographic bounds of the mask
        graph: Optional road graph for binding
        config: Geometry configuration
        
    Returns:
        Dictionary with geojson, mesh, and metadata
    """
    config = config or GeometryConfig()
    
    # Initialize components
    vectorizer = Vectorizer(config)
    mesh_builder = MeshBuilder()
    
    # Extract polygons in pixel coordinates
    polygons_px = vectorizer.mask_to_polygons(mask)
    
    if not polygons_px:
        return {
            "geojson": {"type": "FeatureCollection", "features": []},
            "mesh": {"type": "MeshCollection", "features": [], "count": 0},
            "metadata": {"polygon_count": 0}
        }
    
    # Transform to geographic coordinates
    pixel_size = (mask.shape[1], mask.shape[0])
    polygons_geo = vectorizer.polygons_to_geo(polygons_px, bounds, pixel_size)
    
    # Bind to graph if available
    properties = None
    if graph is not None:
        binder = GraphBinder(graph, config.snap_tolerance)
        properties = binder.bind_polygons(polygons_geo)
    
    # Generate outputs
    geojson = vectorizer.to_geojson(polygons_geo, properties)
    mesh = mesh_builder.polygons_to_mesh_collection(polygons_geo, properties)
    
    return {
        "geojson": geojson,
        "mesh": mesh,
        "metadata": {
            "polygon_count": len(polygons_geo),
            "bounds": bounds.tuple,
            "graph_bound": graph is not None
        }
    }
