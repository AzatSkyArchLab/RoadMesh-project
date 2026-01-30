#!/usr/bin/env python
"""
Visualize road masks and predictions on an interactive map.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RoadMesh Visualization</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        .legend {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}
        .legend h4 {{ margin: 0 0 10px 0; }}
        .legend-item {{ margin: 5px 0; }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 10px;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 16);
        
        var satellite = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{ attribution: 'Esri', maxZoom: 19 }}
        ).addTo(map);
        
        var osm = L.tileLayer(
            'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
            {{ attribution: 'OpenStreetMap', maxZoom: 19 }}
        );
        
        var groundTruth = {ground_truth_geojson};
        var gtLayer = L.geoJSON(groundTruth, {{
            style: {{
                color: '#00ff00',
                weight: 3,
                opacity: 0.9
            }}
        }}).addTo(map);
        
        var predictions = {predictions_geojson};
        var predLayer = L.geoJSON(predictions, {{
            style: {{
                color: '#ff0000',
                weight: 2,
                opacity: 0.8,
                fillColor: '#ff0000',
                fillOpacity: 0.4
            }}
        }});
        
        var bbox = L.rectangle([[{miny}, {minx}], [{maxy}, {maxx}]], {{
            color: '#0000ff',
            weight: 3,
            fill: false
        }}).addTo(map);
        
        var baseMaps = {{ "Satellite": satellite, "OpenStreetMap": osm }};
        var overlays = {{
            "Ground Truth (green)": gtLayer,
            "Predictions (red)": predLayer,
            "Bounding Box": bbox
        }};
        L.control.layers(baseMaps, overlays).addTo(map);
        
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4>RoadMesh</h4>' +
                '<div class="legend-item"><span class="legend-color" style="background:#00ff00"></span>Ground Truth</div>' +
                '<div class="legend-item"><span class="legend-color" style="background:#ff0000"></span>Predictions</div>';
            return div;
        }};
        legend.addTo(map);
        
        map.fitBounds(bbox.getBounds());
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=str, required=True)
    parser.add_argument("--vector-path", type=Path, default=Path("data/geojson_data/osi_sush.geojson"))
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/visualization/map.html"))
    args = parser.parse_args()
    
    bbox = tuple(float(x) for x in args.bbox.split(","))
    minx, miny, maxx, maxy = bbox
    
    print(f"Loading vectors from {args.vector_path}...")
    gdf = gpd.read_file(args.vector_path, bbox=bbox)
    
    if not gdf.empty:
        bbox_geom = box(*bbox)
        gdf = gdf[gdf.intersects(bbox_geom)]
    
    gt_geojson = json.loads(gdf.to_json()) if not gdf.empty else {"type": "FeatureCollection", "features": []}
    print(f"  Found {len(gt_geojson['features'])} road segments")
    
    pred_geojson = {"type": "FeatureCollection", "features": []}
    if args.predictions and args.predictions.exists():
        with open(args.predictions) as f:
            pred_geojson = json.load(f)
        print(f"  Loaded {len(pred_geojson['features'])} predictions")
    
    html = HTML_TEMPLATE.format(
        center_lat=(miny + maxy) / 2,
        center_lon=(minx + maxx) / 2,
        minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        ground_truth_geojson=json.dumps(gt_geojson),
        predictions_geojson=json.dumps(pred_geojson),
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()