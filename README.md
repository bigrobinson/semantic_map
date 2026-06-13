# Metric-Semantic Memory System for Robot LLM Agents

A hierarchical, mutable metric-semantic memory system designed for LLM agents controlling robots (such as from a harness like OpenClaw). The memory system transforms 2D costmaps (occupancy grids) and their accompanying COCO 1.0 JSON annotations into a single, unified JSON store. It supports spatial-semantic queries and automates dynamic memory updates.

---

## 🌟 Features

- **COCO 1.0 Conversion:** Easily import CVAT COCO annotations into a structured, hierarchical JSON memory store.
- **Multimodal AI Integration:** 
  - Generates 768-dimensional text embeddings using Gemini's `gemini-embedding-2` model.
  - Generates detailed semantic descriptions of visual inputs using `gemini-2.5-flash`.
- **Local Embedding Cache:** Employs a local `.embedding_cache.json` file to cache embeddings, maximizing performance and saving API quota.
- **Offline Mock Fallback:** Automatically falls back to deterministic, hash-based mock embeddings if `GEMINI_API_KEY` is not present, allowing offline development and testing.
- **Dynamic Memory Operations:**
  - **Upward Semantic Floating:** Common semantic strings in a region's polygons float up to the region level.
  - **Subregion Proliferation (Splitting):** Groupings of polygons that acquire new shared semantics diverge and split into new subregions.
  - **Region Merging:** Regions that acquire similar meanings (Jaccard similarity >= threshold) automatically merge.
- **Comprehensive CLI:** Includes commands to import, query spatially, query semantically, and add observations on the fly.

---

## 📂 Project Structure

```
.
├── SKILL.md                 # Agent skill specification
├── README.md                # Project documentation
├── annotation_manager.py    # Main memory store & dynamic update manager
├── map_spatial.py           # Spatial query & geometry engine (Shapely)
├── models.py                # Gemini API client & cache logic
├── set_operations.py        # Semantic set logic (Cosine similarity, Union, Diff)
├── utils.py                 # File & JSON schema validation helpers
├── schemas/
│   └── semantic_map_schema.json  # Semantic Map validation schema
├── tests/
│   ├── sample_coco.json     # Test COCO input
│   └── test_memory.py       # Pytest unit tests
└── semantic_map_cli.py      # Executable CLI interface
```

---

## 🛠️ Setup & Installation

The memory system relies on a local virtual environment to manage dependencies.

1. **Create the Virtual Environment:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activate and Install Dependencies:**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   # Or install packages directly:
   pip install shapely pydantic jsonschema requests numpy scipy pillow pytest
   ```

3. **Set API Key (Optional for production):**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

---

## 🚀 CLI Usage

Run the CLI using the python interpreter inside your virtual environment:

### 1. Import COCO Annotations
Converts COCO 1.0 JSON map annotations into the Semantic Map JSON format.
```bash
.venv/bin/python semantic_map_cli.py import-coco \
  --coco-json tests/sample_coco.json \
  --output semantic_map.json \
  --sem-thresh 0.9
```

### 2. Spatial Query
Query the map to retrieve polygons and regions containing a specific coordinate `(x, y)`.
```bash
.venv/bin/python semantic_map_cli.py query-spatial \
  --map-json semantic_map.json \
  -x 150 -y 150
```

### 3. Semantic Search
Perform a semantic search across the map's annotations (polygons and regions) ranked by cosine similarity of text embeddings.
```bash
.venv/bin/python semantic_map_cli.py query-semantic \
  --map-json semantic_map.json \
  --query "chair"
```

### 4. Record an Observation
Add strings or images at coordinate `(x, y)`. This command triggers dynamic memory updates (floating, splitting, and merging) and writes the updated state back to the JSON store.
```bash
.venv/bin/python semantic_map_cli.py add-observation \
  --map-json semantic_map.json \
  -x 450 -y 450 \
  -s "microwave oven" "toaster" \
  --sem-thresh 0.8
```

---

## 🧪 Testing

Run unit tests via `pytest` inside the virtual environment:
```bash
.venv/bin/pytest tests/test_memory.py
```
All tests verify the full suite of transformations, spatial indexes, semantic similarities, and dynamic update loops.
