import os
import sys
import shutil
import pytest
from shapely.geometry import Point

# Ensure the parent directory is in PATH so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation_manager import AnnotationManager
from map_spatial import MapSpatial
from set_operations import set_union, set_intersection, set_difference
from utils import validate_json_schema

COCO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_coco.json")
SCHEMA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas", "semantic_map_schema.json")

@pytest.fixture
def temp_coco_path(tmp_path):
    """Fixture that copies the sample COCO file to a temporary location for each test."""
    temp_file = tmp_path / "temp_coco.json"
    shutil.copy(COCO_FILE, temp_file)
    return str(temp_file)

def test_coco_import_and_schema(temp_coco_path, tmp_path):
    output_map = str(tmp_path / "semantic_map.json")
    manager = AnnotationManager(json_file=temp_coco_path, map_file="occupancy_grid.png")
    
    # Save output map
    manager.write_json(output_map)
    assert os.path.exists(output_map)
    
    # Validate against our custom schema
    valid = validate_json_schema(SCHEMA_FILE, manager._get_serialized_data())
    assert valid, "Generated Semantic Map JSON does not conform to the schema."

def test_spatial_queries(temp_coco_path):
    manager = AnnotationManager(json_file=temp_coco_path)
    
    # Point (150, 150) should be inside polygon 0 and polygon 1 (Desk Workspace)
    polys = manager.get_polygons_point(150.0, 150.0)
    assert 0 in polys
    assert 1 in polys
    
    # Centroid of polygon 0 should be at (150, 150)
    centroid = manager.map_spatial.get_polygon_centroid(0)
    assert centroid.x == 150.0
    assert centroid.y == 150.0

def test_semantic_similarity_and_search(temp_coco_path):
    manager = AnnotationManager(json_file=temp_coco_path)
    
    # Search for "monitor" which is in polygon 0 ("computer monitor")
    query_emb = manager.map_spatial.data['polygons'][0]['strings'] # list of raw strings
    assert "computer monitor" in query_emb

def test_float_up_strings(temp_coco_path):
    manager = AnnotationManager(json_file=temp_coco_path)
    
    # Manually add a common string to both polygons of "Desk Workspace" (region 1)
    # Desk Workspace contains polygons 0 and 1
    manager.add_string_polygon(0, "office furniture", sem_thresh=0.7)
    manager.add_string_polygon(1, "office furniture", sem_thresh=0.7)
    
    # Trigger floating up
    manager.float_up_strings(sem_thresh=0.7)
    
    # The common string should now be in the region strings
    region_1 = next((r for r in manager.data['regions'] if r['id'] == 1), None)
    assert "office furniture" in region_1['strings']
    
    # And removed from the individual polygons
    poly_0 = next((p for p in manager.data['polygons'] if p['id'] == 0), None)
    poly_1 = next((p for p in manager.data['polygons'] if p['id'] == 1), None)
    assert "office furniture" not in poly_0['strings']
    assert "office furniture" not in poly_1['strings']

def test_subregion_proliferation(temp_coco_path):
    manager = AnnotationManager(json_file=temp_coco_path)
    
    # Region 1 (Desk Workspace) contains polygon 0 and 1.
    # Let's say polygon 0 and 1 are in Region 1, and we add an extra polygon 3 to Region 1
    new_poly = {
        "id": 3,
        "region_ids": [1],
        "polygon": [300.0, 300.0, 400.0, 300.0, 400.0, 400.0, 300.0, 400.0],
        "polygon_centroid": [350.0, 350.0],
        "strings": ["chair"],
        "images": []
    }
    manager.data['polygons'].append(new_poly)
    
    # Reindex and set polygon_ids mapping
    manager._reindex_polygons()
    manager._add_polygon_ids_to_regions()
    
    # Desk Workspace now has polygons 0, 1, and 3.
    # Let's add new strings in common specifically to polygons 0 and 1 (a plural subset of size 2)
    manager.add_string_polygon(0, "ergonomic office stuff", sem_thresh=0.7)
    manager.add_string_polygon(1, "ergonomic office stuff", sem_thresh=0.7)
    
    # Base region strings for region 1
    region_1 = next((r for r in manager.data['regions'] if r['id'] == 1), None)
    region_1['strings'] = ["desk"]
    
    # Trigger subregion splitting
    # Since "ergonomic office stuff" diverges semantically from "desk", Jaccard distance will be high
    manager.proliferate_subregions(sem_thresh=0.7, split_threshold=0.5)
    
    # A new region (id=3) should be created as a subregion of region 1
    new_reg = next((r for r in manager.data['regions'] if r['id'] == 3), None)
    assert new_reg is not None
    assert "ergonomic office stuff" in new_reg['strings']
    assert 1 in new_reg['superregions']
    assert 3 in region_1['subregions']

def test_region_merging(temp_coco_path):
    manager = AnnotationManager(json_file=temp_coco_path)
    
    # Let's set the strings of region 1 and 2 to be identical
    reg_1 = next((r for r in manager.data['regions'] if r['id'] == 1), None)
    reg_2 = next((r for r in manager.data['regions'] if r['id'] == 2), None)
    
    reg_1['strings'] = ["workspace setup"]
    reg_2['strings'] = ["workspace setup"]
    
    # Trigger merging
    manager.merge_regions(sem_thresh=0.7, merge_threshold=0.7)
    
    # One of the regions should be deleted and combined into the other
    ids = [r['id'] for r in manager.data['regions']]
    assert len(ids) == 2 # Originally 3 regions (0, 1, 2), now 2 regions after merge

def test_add_observation(temp_coco_path):
    manager = AnnotationManager(json_file=temp_coco_path)
    
    # Point (450, 450) is inside polygon 2 (Kitchen Counter)
    # Let's add an observation here
    manager.add_observation(
        x=450.0,
        y=450.0,
        strings=["microwave appliance", "kitchen counter"],
        sem_thresh=0.7,
        split_thresh=0.7,
        merge_thresh=0.7
    )
    
    # Verify the observation was appended (either in polygon or floated up to its region)
    poly_2 = next((p for p in manager.data['polygons'] if p['id'] == 2), None)
    reg_2 = next((r for r in manager.data['regions'] if 2 in r.get('polygon_ids', [])), None)
    assert any("microwave" in s for s in poly_2['strings']) or any("microwave" in s for s in reg_2['strings'])
