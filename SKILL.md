---
name: metric-semantic-memory
description: >-
  A metric-semantic memory system for a robot or LLM agent. Converts COCO annotations 
  into a single hierarchical JSON store, enabling spatial-semantic queries and 
  dynamic memory updates like upward semantic floating, subregion splitting, and region merging.
---

# Metric-Semantic Memory System Skill

## Overview
This skill provides a hierarchical, mutable metric-semantic memory system for robots controlled by LLM agents (e.g. running in harnesses like OpenClaw). It manages a single JSON map store containing spatial polygons (with centroids and optional image descriptors) and regions. It supports spatial queries, semantic search, and automatic spatio-semantic updates (such as floating common names up to regions, splitting subregions when attributes diverge, and merging regions when meanings align).

## Dependencies
- `python3` (>= 3.10)
- `shapely` (for 2D geometry operations)
- `pydantic` (for parsing and structures)
- `jsonschema` (for validating map files)
- `scipy` / `numpy` (for cosine similarity and centroids)
- `requests` (for calling the Gemini API to get embeddings)

## Quick Start
To import a COCO annotation map and query it:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Import COCO 1.0 JSON map
python semantic_map_cli.py import-coco --coco-json sample_coco.json --output semantic_map.json

# Query the map for things near coordinates (x=150, y=200)
python semantic_map_cli.py query-spatial --map-json semantic_map.json -x 150.0 -y 200.0

# Conduct a semantic search for a charging station
python semantic_map_cli.py query-semantic --map-json semantic_map.json --query "battery charging station"
```

## Utility Scripts

The skill exposes the `semantic_map_cli.py` utility script with four subcommands:

### 1. `import-coco`
Transforms standard CVAT COCO 1.0 annotations to our custom Semantic Map JSON format.
```bash
python semantic_map_cli.py import-coco \
  --coco-json <path_to_coco_json> \
  --output <path_to_save_semantic_map_json> \
  [--map-image <path_to_map_png>] \
  [--coco-schema <path_to_coco_schema>] \
  [--sem-thresh 0.7]
```

### 2. `query-spatial`
Queries regions and polygons enclosing a point `(x, y)`.
```bash
python semantic_map_cli.py query-spatial \
  --map-json <path_to_semantic_map_json> \
  -x <x_coordinate> \
  -y <y_coordinate> \
  [--output <path_to_save_results_json>]
```

### 3. `query-semantic`
Conducts a semantic search against strings in regions and polygons. Ranked by cosine similarity of text embeddings.
```bash
python semantic_map_cli.py query-semantic \
  --map-json <path_to_semantic_map_json> \
  --query "text query to search" \
  [-k <top_k_results>] \
  [--output <path_to_save_results_json>]
```

### 4. `add-observation`
Records a semantic observation at `(x, y)`. If no polygon exists there, a small 10x10 polygon is created. The observation is appended and dynamic memory updates (upward floating, subregion splitting, and region merging) are executed automatically.
```bash
python semantic_map_cli.py add-observation \
  --map-json <path_to_semantic_map_json> \
  -x <x_coordinate> \
  -y <y_coordinate> \
  -s "first attribute" "second attribute" \
  [--image-url <url_to_image_if_observed>] \
  [--sem-thresh 0.7] \
  [--split-thresh 0.7] \
  [--merge-thresh 0.7] \
  [--output <path_to_save_summary_json>]
```

## Rate Limiting
- Embedding generation uses the `gemini-embedding-2` API and image descriptions use `gemini-2.5-flash`.
- **Local Embedding Cache:** An internal cache is saved to `.embedding_cache.json` automatically, minimizing API calls and maximizing local execution speed.
- If `GEMINI_API_KEY` is not present, the CLI falls back to deterministic mock embeddings, which prevents crashes and rate-limit hits during testing or offline runs.

## Common Mistakes
1. **Executing outside the Virtual Environment:** Always run the commands inside the virtual environment (`source .venv/bin/activate`) or call the interpreter directly (`.venv/bin/python`).
2. **Invalid Coordinates:** X and Y coordinates must correspond to the pixel dimensions of the base occupancy map. Check the `width` and `height` inside `map_info` to ensure coordinates are in range.
3. **Missing API Key:** If running in production with real sensor inputs, make sure the `GEMINI_API_KEY` is set in your environment.
