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
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
import geopandas as gpd

from roadmesh.core.config import GeometryConfig, BBox


class Vectorizer:
    """
    Converts binary segmentation masks to vector polygons.
    """

    def __init__(self, config: Optional[GeometryConfig] = None):
        self.config = config or GeometryConfig()

    def cleanup_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        iterations: int = 1
    ) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations
        )
        return mask

    def extract_contours(self, mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_L1
        )
        return contours

    def _extract_polygons(self, geom) -> list[Polygon]:
        """Extract only valid Polygon objects from any geometry."""
        result = []
        if geom is None or geom.is_empty:
            return result
        if isinstance(geom, Polygon):
            if geom.area >= self.config.min_polygon_area:
                result.append(geom)
        elif isinstance(geom, (MultiPolygon, GeometryCollection)):
            for g in geom.geoms:
                result.extend(self._extract_polygons(g))
        return result

    def contour_to_polygon(self, contour: np.ndarray, simplify: bool = True) -> list[Polygon]:
        if len(contour) < 3:
            return []
        coords = contour.squeeze()
        if coords.ndim == 1:
            return []
        try:
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            polygons = self._extract_polygons(polygon)
            if simplify:
                simplified = []
                for p in polygons:
                    s = p.simplify(self.config.simplify_tolerance, preserve_topology=True)
                    simplified.extend(self._extract_polygons(s))
                polygons = simplified
            return polygons
        except Exception:
            return []

    def mask_to_polygons(
        self,
        mask: np.ndarray,
        cleanup: bool = True,
        simplify: bool = True
    ) -> list[Polygon]:
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        mask = (mask > 127).astype(np.uint8) * 255
        if cleanup:
            mask = self.cleanup_mask(mask)
        contours = self.extract_contours(mask)
        polygons = []
        for contour in contours:
            polys = self.contour_to_polygon(contour, simplify=simplify)
            polygons.extend(polys)
        return polygons

    def polygons_to_geo(
        self,
        polygons: list[Polygon],
        bounds: BBox,
        pixel_size: tuple[int, int]
    ) -> list[Polygon]:
        width, height = pixel_size
        x_scale = (bounds.maxx - bounds.minx) / width
        y_scale = (bounds.maxy - bounds.miny) / height
        geo_polygons = []
        for polygon in polygons:
            try:
                # Skip non-Polygon types
                if not isinstance(polygon, Polygon):
                    continue
                if polygon.is_empty:
                    continue
                if polygon.exterior is None:
                    continue
                    
                coords = np.array(polygon.exterior.coords)
                geo_coords = np.column_stack([
                    bounds.minx + coords[:, 0] * x_scale,
                    bounds.maxy - coords[:, 1] * y_scale
                ])
                geo_polygon = Polygon(geo_coords)
                if geo_polygon.is_valid and not geo_polygon.is_empty:
                    geo_polygons.append(geo_polygon)
                else:
                    valid = make_valid(geo_polygon)
                    geo_polygons.extend(self._extract_polygons(valid))
            except Exception:
                continue
        return geo_polygons

    def to_geojson(
        self,
        polygons: list[Polygon],
        properties: Optional[list[dict]] = None
    ) -> dict:
        features = []
        for i, polygon in enumerate(polygons):
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
            props = properties[i] if properties and i < len(properties) else {}
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {"id": i, **props}
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}


class MeshBuilder:
    def __init__(self, default_elevation: float = 0.0):
        self.default_elevation = default_elevation

    def triangulate_polygon(self, polygon: Polygon) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return np.array([]), np.array([])
        if polygon.exterior is None:
            return np.array([]), np.array([])
        coords = np.array(polygon.exterior.coords)[:-1]
        n_vertices = len(coords)
        if n_vertices < 3:
            return np.array([]), np.array([])
        vertices = coords.copy()
        indices = []
        if polygon.is_valid and self._is_convex(coords):
            for i in range(1, n_vertices - 1):
                indices.extend([0, i, i + 1])
        else:
            centroid = np.array(polygon.centroid.coords[0])
            vertices = np.vstack([vertices, centroid])
            center_idx = n_vertices
            for i in range(n_vertices):
                next_i = (i + 1) % n_vertices
                indices.extend([center_idx, i, next_i])
        return vertices, np.array(indices, dtype=np.int32)

    def _is_convex(self, coords: np.ndarray) -> bool:
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
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return None
        elevation = elevation if elevation is not None else self.default_elevation
        vertices_2d, indices = self.triangulate_polygon(polygon)
        if len(vertices_2d) == 0:
            return None
        n_vertices = len(vertices_2d)
        positions = np.zeros((n_vertices, 3), dtype=np.float32)
        positions[:, :2] = vertices_2d
        positions[:, 2] = elevation
        normals = np.zeros((n_vertices, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        bounds = polygon.bounds
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
        features = []
        for i, polygon in enumerate(polygons):
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
            props = properties_list[i] if properties_list and i < len(properties_list) else {"id": i}
            mesh = self.polygon_to_mesh(polygon, elevation, props)
            if mesh is not None:
                features.append(mesh)
        return {"type": "MeshCollection", "features": features, "count": len(features)}

    def to_threejs_json(
        self,
        polygons: list[Polygon],
        properties_list: Optional[list[dict]] = None,
    ) -> str:
        collection = self.polygons_to_mesh_collection(polygons, properties_list)
        return json.dumps(collection)


class GraphBinder:
    def __init__(
        self,
        graph: Optional[gpd.GeoDataFrame] = None,
        snap_tolerance: float = 5.0
    ):
        self.graph = graph
        self.snap_tolerance = snap_tolerance

    def load_graph(self, path: str) -> None:
        self.graph = gpd.read_file(path)

    def find_nearest_edge(self, polygon: Polygon) -> Optional[tuple[int, dict]]:
        if self.graph is None or self.graph.empty:
            return None
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            return None
        centroid = polygon.centroid
        distances = self.graph.geometry.distance(centroid)
        min_idx = distances.idxmin()
        min_dist = distances[min_idx]
        if min_dist > self.snap_tolerance:
            return None
        edge = self.graph.loc[min_idx]
        properties = edge.drop('geometry').to_dict()
        return min_idx, properties

    def bind_polygons(self, polygons: list[Polygon]) -> list[dict]:
        bindings = []
        for polygon in polygons:
            result = self.find_nearest_edge(polygon)
            if result is not None:
                edge_idx, edge_props = result
                bindings.append({"bound": True, "edge_id": int(edge_idx), **edge_props})
            else:
                bindings.append({"bound": False, "edge_id": None})
        return bindings


def mask_to_mesh(
    mask: np.ndarray,
    bounds: BBox,
    graph: Optional[gpd.GeoDataFrame] = None,
    config: Optional[GeometryConfig] = None
) -> dict:
    config = config or GeometryConfig()
    vectorizer = Vectorizer(config)
    mesh_builder = MeshBuilder()
    polygons_px = vectorizer.mask_to_polygons(mask)
    if not polygons_px:
        return {
            "geojson": {"type": "FeatureCollection", "features": []},
            "mesh": {"type": "MeshCollection", "features": [], "count": 0},
            "metadata": {"polygon_count": 0}
        }
    pixel_size = (mask.shape[1], mask.shape[0])
    polygons_geo = vectorizer.polygons_to_geo(polygons_px, bounds, pixel_size)
    properties = None
    if graph is not None:
        binder = GraphBinder(graph, config.snap_tolerance)
        properties = binder.bind_polygons(polygons_geo)
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