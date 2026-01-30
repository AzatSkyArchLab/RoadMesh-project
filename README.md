# RoadMesh 🛣️

ML-система для извлечения mesh-геометрии дорог из спутниковых снимков.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Возможности

- 🛰️ Автоматическая загрузка спутниковых тайлов (Esri World Imagery)
- 🧠 Сегментация дорог с помощью D-LinkNet / U-Net
- 🔷 Векторизация масок в GeoJSON полигоны
- 🎮 Экспорт в Three.js mesh формат
- 🔗 Привязка к дорожному графу
- 🚀 REST API для inference

## Быстрый старт

### Установка

```bash
# Клонирование
git clone https://github.com/YOUR_USERNAME/roadmesh.git
cd roadmesh

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или .venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -e ".[dev]"
```

### Требования

- Python 3.10+
- CUDA 11.8+ (для GPU обучения)
- **GPU: 6-8+ GB VRAM** (RTX 3060/3070/3080)
  - RTX 3070 Ti 8GB: batch_size=4 с mixed precision ✓

### Тестирование пайплайна

```bash
# Быстрый тест на небольшой области Москвы
python scripts/test_pipeline.py --area kremlin

# Или с кастомным bbox
python scripts/test_pipeline.py --bbox 37.61,55.74,37.63,55.76
```

## Использование

### 1. Подготовка датасета

```bash
# Положи свои векторные данные дорог в data/raw/
# Поддерживаются: GeoJSON, Shapefile, GeoPackage

python scripts/prepare_dataset.py \
  --vector-path data/raw/moscow_roads.geojson \
  --bbox 37.35,55.55,37.85,55.95 \
  --output-dir data/processed \
  --zoom 18
```

### 2. Обучение модели

```bash
# С дефолтным конфигом (оптимизирован для 8GB VRAM)
python scripts/train.py

# С кастомными параметрами
python scripts/train.py \
  --config configs/model/dlinknet_8gb.yaml \
  --epochs 50 \
  --batch-size 4

# Debug режим (быстрая проверка)
python scripts/train.py --debug
```

### 3. Inference

```bash
# Предсказание на новой области
python scripts/predict.py \
  --checkpoint checkpoints/best_model.pt \
  --bbox 37.6,55.75,37.65,55.78 \
  --output results/
```

### 4. Запуск API

```bash
# Локальный сервер
python -m roadmesh.api.app

# Или с uvicorn
uvicorn roadmesh.api.app:app --host 0.0.0.0 --port 8000 --reload
```

API будет доступен по адресу: http://localhost:8000

Документация: http://localhost:8000/docs

## API Reference

### POST /api/v1/predict

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": [37.61, 55.74, 37.63, 55.76],
    "zoom": 18,
    "output_format": "both"
  }'
```

Response:
```json
{
  "geojson": { "type": "FeatureCollection", "features": [...] },
  "mesh": { "type": "MeshCollection", "features": [...] },
  "metadata": { "polygon_count": 42, "bbox": [...] }
}
```

## Структура проекта

```
roadmesh/
├── configs/              # YAML конфигурации
│   ├── base.yaml
│   ├── data/moscow.yaml
│   └── model/dlinknet_8gb.yaml
├── data/                 # Данные (gitignored)
│   ├── raw/              # Исходные тайлы + векторы
│   ├── processed/        # Готовый датасет
│   └── cache/            # Кэш тайлов
├── scripts/              # CLI скрипты
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── test_pipeline.py
├── src/roadmesh/         # Основной код
│   ├── core/             # Конфигурация, типы
│   ├── data/             # Загрузка данных
│   ├── models/           # Архитектуры нейросетей
│   ├── training/         # Training loop
│   ├── inference/        # Предсказание
│   ├── geometry/         # Векторизация, mesh
│   └── api/              # FastAPI
└── notebooks/            # Jupyter эксперименты
```

## Конфигурация

### Оптимизация для 8GB VRAM

```yaml
# configs/model/dlinknet_8gb.yaml
training:
  batch_size: 4
  gradient_accumulation: 2  # Effective batch = 8
  mixed_precision: true     # FP16, экономит ~40% VRAM
```

### Esri World Imagery

Бесплатный провайдер спутниковых снимков:
- Разрешение: ~0.5м на zoom 18
- Ограничения: rate limiting (~1 req/sec)
- Лицензия: некоммерческое использование

## Интеграция с Insol Web

```javascript
// JavaScript клиент
const roadmesh = new RoadMeshClient('https://api.your-domain.com');

async function loadRoads(bbox) {
  const result = await roadmesh.predict(bbox);
  
  // Создание Three.js mesh
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', 
    new THREE.Float32BufferAttribute(result.mesh.positions, 3));
  
  const mesh = new THREE.Mesh(geometry, roadMaterial);
  scene.add(mesh);
}
```

## Лицензия

MIT

## Авторы

- Azat Foxie
