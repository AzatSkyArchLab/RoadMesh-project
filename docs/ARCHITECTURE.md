# RoadMesh: Архитектура проекта

## Обзор

**RoadMesh** — модульная ML-система для извлечения mesh-геометрии дорог из спутниковых снимков с привязкой к дорожному графу. Проектируется как переиспользуемая библиотека + API-сервис.

---

## Системная архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROADMESH SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   DATA      │  │   MODELS    │  │  GEOMETRY   │  │      API            │ │
│  │   MODULE    │  │   MODULE    │  │  MODULE     │  │      MODULE         │ │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────────────┤ │
│  │ • Tile      │  │ • Segment-  │  │ • Vectori-  │  │ • FastAPI           │ │
│  │   Fetcher   │  │   ation     │  │   zation    │  │   endpoints         │ │
│  │ • Label     │  │ • Graph     │  │ • Graph     │  │ • WebSocket         │ │
│  │   Processor │  │   Extraction│  │   Binding   │  │   streaming         │ │
│  │ • Dataset   │  │ • Training  │  │ • Mesh      │  │ • Tile cache        │ │
│  │   Builder   │  │   Pipeline  │  │   Export    │  │ • Auth              │ │
│  │ • Augment   │  │ • Inference │  │ • CRS       │  │ • Rate limiting     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                │                    │            │
│         └────────────────┼────────────────┼────────────────────┘            │
│                          ▼                ▼                                  │
│                   ┌─────────────────────────────┐                           │
│                   │        CORE                  │                           │
│                   ├─────────────────────────────┤                           │
│                   │ • Config (Pydantic)          │                           │
│                   │ • Logging (structlog)        │                           │
│                   │ • Exceptions                 │                           │
│                   │ • Types & Protocols          │                           │
│                   └─────────────────────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL INTEGRATIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Esri World Imagery  │  Custom Vector Data  │  Insol Web  │  Other Clients  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Структура проекта

```
roadmesh/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Тесты, линтинг на PR
│       ├── release.yml            # Сборка Docker, публикация
│       └── train.yml              # GitHub Actions для обучения (опционально)
│
├── configs/
│   ├── base.yaml                  # Базовая конфигурация
│   ├── data/
│   │   ├── moscow.yaml            # Конфиг датасета Москва
│   │   └── custom.yaml            # Шаблон для своих данных
│   ├── model/
│   │   ├── dlinknet.yaml          # D-LinkNet конфиг
│   │   ├── unet_resnet34.yaml     # UNet + ResNet34
│   │   └── lightweight.yaml       # Легкая модель для 8GB VRAM
│   └── deploy/
│       ├── local.yaml             # Локальный деплой
│       └── production.yaml        # Продакшн настройки
│
├── data/
│   ├── raw/                       # Сырые данные (gitignore)
│   │   ├── tiles/                 # Спутниковые тайлы
│   │   └── vectors/               # Твои векторные данные
│   ├── processed/                 # Обработанный датасет
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── cache/                     # Кэш тайлов API
│
├── src/
│   └── roadmesh/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py          # Pydantic конфигурации
│       │   ├── logging.py         # Настройка логирования
│       │   ├── exceptions.py      # Кастомные исключения
│       │   └── types.py           # Type aliases, protocols
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── tile_fetcher.py    # Загрузка спутниковых тайлов
│       │   ├── label_processor.py # Обработка векторных меток
│       │   ├── dataset.py         # PyTorch Dataset
│       │   ├── augmentations.py   # Albumentations пайплайн
│       │   └── builder.py         # Сборка датасета
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py            # Базовый класс модели
│       │   ├── dlinknet.py        # D-LinkNet архитектура
│       │   ├── unet.py            # UNet варианты
│       │   ├── losses.py          # Loss functions
│       │   └── metrics.py         # IoU, APLS, connectivity
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py         # Training loop
│       │   ├── callbacks.py       # Checkpointing, early stopping
│       │   └── scheduler.py       # LR scheduling
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predictor.py       # Inference pipeline
│       │   ├── stitcher.py        # Сшивка больших областей
│       │   └── onnx_export.py     # Экспорт в ONNX
│       │
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── vectorizer.py      # Mask → Polygon
│       │   ├── graph_binding.py   # Привязка к графу дорог
│       │   ├── mesh_builder.py    # Polygon → Mesh
│       │   └── crs.py             # Координатные преобразования
│       │
│       └── api/
│           ├── __init__.py
│           ├── app.py             # FastAPI приложение
│           ├── routes/
│           │   ├── __init__.py
│           │   ├── predict.py     # /predict endpoint
│           │   ├── tiles.py       # /tiles proxy/cache
│           │   └── health.py      # Health checks
│           ├── schemas.py         # Pydantic request/response
│           └── dependencies.py    # DI для модели и конфигов
│
├── scripts/
│   ├── prepare_dataset.py         # CLI: создание датасета
│   ├── train.py                   # CLI: обучение
│   ├── evaluate.py                # CLI: оценка модели
│   ├── predict.py                 # CLI: inference
│   └── export_onnx.py             # CLI: экспорт модели
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_data/
│   │   ├── test_models/
│   │   └── test_geometry/
│   ├── integration/
│   │   └── test_pipeline.py
│   └── fixtures/
│       └── sample_data/
│
├── docker/
│   ├── Dockerfile                 # Production image
│   ├── Dockerfile.dev             # Development image
│   └── docker-compose.yml         # Local stack
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_experiments.ipynb
│   └── 03_inference_demo.ipynb
│
├── docs/
│   ├── ARCHITECTURE.md            # Этот документ
│   ├── SETUP.md                   # Инструкция по установке
│   ├── TRAINING.md                # Гайд по обучению
│   ├── API.md                     # API документация
│   └── INTEGRATION.md             # Интеграция с Insol
│
├── .env.example                   # Шаблон переменных окружения
├── .gitignore
├── .pre-commit-config.yaml        # Pre-commit hooks
├── pyproject.toml                 # Poetry/pip конфигурация
├── Makefile                       # Команды для разработки
└── README.md
```

---

## Модули детально

### 1. Data Module

```python
# Основные компоненты

class TileFetcher:
    """Загрузка спутниковых тайлов с кэшированием"""
    
    providers = {
        'esri': EsriWorldImagery,      # Бесплатный, хорошее качество
        'mapbox': MapboxSatellite,     # Требует API key
        'bing': BingAerial,            # Бесплатный лимит
    }
    
    def fetch_tile(self, x: int, y: int, z: int) -> np.ndarray
    def fetch_bbox(self, bbox: BBox, zoom: int) -> np.ndarray
    def get_cached(self, tile_id: str) -> Optional[np.ndarray]


class LabelProcessor:
    """Конвертация векторных данных в маски"""
    
    def load_vectors(self, path: Path) -> gpd.GeoDataFrame
    def rasterize(self, gdf: gpd.GeoDataFrame, shape: tuple) -> np.ndarray
    def create_distance_transform(self, mask: np.ndarray) -> np.ndarray
    def create_orientation_field(self, mask: np.ndarray) -> np.ndarray


class RoadDataset(torch.utils.data.Dataset):
    """PyTorch Dataset с аугментациями"""
    
    def __init__(self, 
                 tiles_dir: Path,
                 masks_dir: Path,
                 transform: A.Compose = None)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]
```

### 2. Models Module

```python
# Архитектуры оптимизированные для 8GB VRAM

class DLinkNet34(nn.Module):
    """D-LinkNet с ResNet34 encoder
    
    Memory: ~3GB при batch_size=4, input 512×512
    """
    
    def __init__(self, num_classes: int = 1, pretrained: bool = True)
    def forward(self, x: Tensor) -> Tensor


class LightweightUNet(nn.Module):
    """Легкая версия для быстрого прототипирования
    
    Memory: ~1.5GB при batch_size=8, input 512×512
    Backbone: EfficientNet-B0
    """


# Loss functions
class CombinedLoss(nn.Module):
    """BCE + Dice + Connectivity"""
    
    def __init__(self,
                 bce_weight: float = 0.5,
                 dice_weight: float = 0.3,
                 conn_weight: float = 0.2)


class ConnectivityLoss(nn.Module):
    """Penalizes breaks in road connectivity"""
```

### 3. Geometry Module

```python
# Векторизация и создание mesh

class Vectorizer:
    """Mask → Polygon конвертация"""
    
    def __init__(self,
                 simplify_tolerance: float = 1.0,
                 min_area: float = 10.0)
    
    def vectorize(self, mask: np.ndarray) -> list[Polygon]
    def simplify(self, polygons: list[Polygon]) -> list[Polygon]
    def to_geojson(self, polygons: list[Polygon], crs: CRS) -> dict


class GraphBinder:
    """Привязка полигонов к дорожному графу"""
    
    def __init__(self, graph: nx.Graph)
    
    def bind(self, polygon: Polygon) -> dict[str, Any]:
        """Returns edge_id and inherited attributes"""
    
    def snap_to_graph(self, polygon: Polygon) -> Polygon:
        """Snap polygon vertices to nearest graph edges"""


class MeshBuilder:
    """Polygon → 3D Mesh для Three.js"""
    
    def build_mesh(self, 
                   polygon: Polygon,
                   elevation: float = 0.0) -> dict:
        """Returns Three.js BufferGeometry-compatible format"""
        return {
            'positions': [...],  # Float32Array
            'indices': [...],    # Uint32Array  
            'normals': [...],    # Float32Array
            'uvs': [...],        # Float32Array (optional)
        }
```

### 4. API Module

```python
# FastAPI endpoints

# POST /api/v1/predict
class PredictRequest(BaseModel):
    bbox: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    zoom: int = 18
    output_format: Literal['geojson', 'mesh', 'both'] = 'both'
    graph_binding: bool = True

class PredictResponse(BaseModel):
    geojson: Optional[dict]
    mesh: Optional[dict]  # Three.js BufferGeometry format
    metadata: dict


# WebSocket /api/v1/predict/stream
# Для больших областей — стриминг тайлов по мере обработки


# GET /api/v1/tiles/{z}/{x}/{y}
# Прокси к спутниковым тайлам с кэшированием
```

---

## Конфигурация

### configs/base.yaml

```yaml
# Базовая конфигурация проекта

project:
  name: roadmesh
  version: 0.1.0

data:
  tile_size: 512
  tile_provider: esri
  cache_dir: ./data/cache
  
  # Для Москвы
  default_bbox:
    minx: 37.35
    miny: 55.55
    maxx: 37.85
    maxy: 55.95
  
  augmentations:
    horizontal_flip: true
    vertical_flip: true
    rotate_90: true
    brightness_contrast: true
    
model:
  architecture: dlinknet34
  input_size: 512
  num_classes: 1
  pretrained_backbone: true

training:
  batch_size: 4          # Для 8GB VRAM
  num_workers: 4
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.00001
  
  # Mixed precision для экономии памяти
  mixed_precision: true
  gradient_accumulation: 2  # Эффективный batch_size = 8
  
  early_stopping:
    patience: 15
    min_delta: 0.001
    
  scheduler:
    type: cosine
    warmup_epochs: 5
    
inference:
  batch_size: 8
  tile_overlap: 64        # Пиксели перекрытия при сшивке
  
geometry:
  simplify_tolerance: 1.5
  min_polygon_area: 50
  snap_tolerance: 2.0
  
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  tile_cache_size: 1000   # LRU cache
  
logging:
  level: INFO
  format: json
```

---

## Оптимизация для 8GB VRAM

### Стратегии

1. **Mixed Precision Training (FP16)**
   - Экономит ~40% VRAM
   - PyTorch native: `torch.cuda.amp`

2. **Gradient Accumulation**
   - batch_size=4 × accumulation=2 = effective batch 8
   - Стабильнее обучение без увеличения памяти

3. **Gradient Checkpointing**
   - Пересчёт активаций при backprop
   - Экономит ~30% VRAM, +20% время

4. **Efficient Backbone**
   - ResNet34 вместо ResNet50/101
   - EfficientNet-B0 для lightweight версии

### Memory Budget

```
RTX 3070 Ti: 8GB VRAM

Model (ResNet34 encoder):     ~100 MB
Input batch (4 × 512 × 512):  ~12 MB  
Activations + gradients:      ~4-5 GB
Optimizer states (AdamW):     ~200 MB
CUDA overhead:                ~500 MB
─────────────────────────────────────
Total:                        ~5-6 GB ✓

Headroom for peaks:           ~2 GB ✓
```

---

## Deployment

### Local Development

```bash
# 1. Клонирование и установка
git clone https://github.com/your-username/roadmesh.git
cd roadmesh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Подготовка датасета
python scripts/prepare_dataset.py --config configs/data/moscow.yaml

# 3. Обучение
python scripts/train.py --config configs/model/dlinknet.yaml

# 4. Запуск API
python -m roadmesh.api.app
```

### Docker Production

```dockerfile
# docker/Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY pyproject.toml .
RUN pip install .

COPY src/ src/
COPY configs/ configs/
COPY weights/ weights/

EXPOSE 8000
CMD ["uvicorn", "roadmesh.api.app:app", "--host", "0.0.0.0"]
```

### Deploy на твой домен

```yaml
# docker-compose.yml
version: '3.8'

services:
  roadmesh-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data/cache:/app/data/cache
      - ./weights:/app/weights
    environment:
      - ROADMESH_CONFIG=configs/deploy/production.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt
    depends_on:
      - roadmesh-api
```

---

## Интеграция с Insol Web

### JavaScript Client

```javascript
// insol-roadmesh-client.js

class RoadMeshClient {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
  }

  async predict(bbox, options = {}) {
    const response = await fetch(`${this.apiUrl}/api/v1/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox,
        zoom: options.zoom || 18,
        output_format: 'mesh',
        graph_binding: true
      })
    });
    
    return response.json();
  }

  createThreeJsMesh(meshData, material) {
    const geometry = new THREE.BufferGeometry();
    
    geometry.setAttribute('position', 
      new THREE.Float32BufferAttribute(meshData.positions, 3));
    geometry.setIndex(meshData.indices);
    geometry.computeVertexNormals();
    
    return new THREE.Mesh(geometry, material);
  }
}

// Использование в Insol
const roadmesh = new RoadMeshClient('https://api.your-domain.com');

async function loadRoads(bbox) {
  const result = await roadmesh.predict(bbox);
  
  const roadMaterial = new THREE.MeshStandardMaterial({
    color: 0x333333,
    roughness: 0.9
  });
  
  for (const feature of result.mesh.features) {
    const mesh = roadmesh.createThreeJsMesh(feature.geometry, roadMaterial);
    mesh.userData = feature.properties; // edge_id, road_type, etc.
    scene.add(mesh);
  }
}
```

---

## Roadmap

### Phase 1: MVP (4-6 недель)
- [x] Архитектура и проект
- [ ] Data pipeline (Esri + твои векторы)
- [ ] Базовая модель (D-LinkNet)
- [ ] Training на Москве
- [ ] Простой API endpoint

### Phase 2: Production (2-3 недели)
- [ ] Оптимизация модели
- [ ] ONNX export
- [ ] Docker deployment
- [ ] Интеграция с Insol

### Phase 3: Enhancement (ongoing)
- [ ] Graph-aware модель (SAM-Road style)
- [ ] Multi-city training
- [ ] Browser inference (ONNX.js)
- [ ] Real-time updates
