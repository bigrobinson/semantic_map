from __future__ import annotations
import json
import os
import copy
import numpy as np
from shapely.geometry import Point, Polygon
from typing import List, Dict, Set, Any, Tuple
from set_operations import (
    polygon_iou, 
    set_union, 
    set_intersection, 
    set_difference, 
    get_embedding
)
from utils import validate_json_schema, load_json, save_json
from map_spatial import MapSpatial
from models import embedding_model

def semantic_jaccard_similarity(set1: Set[str], set2: Set[str], similarity_threshold: float = 0.7) -> float:
    """Computes Jaccard Similarity between two sets of strings based on semantic equality."""
    if not set1 and not set2:
        return 1.0
    intersection = set_intersection(set1, set2, similarity_threshold=similarity_threshold)
    union = set_union(set1, set2, similarity_threshold=similarity_threshold)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def semantic_jaccard_distance(set1: Set[str], set2: Set[str], similarity_threshold: float = 0.7) -> float:
    """Computes Jaccard Distance between two sets of strings based on semantic equality."""
    return 1.0 - semantic_jaccard_similarity(set1, set2, similarity_threshold=similarity_threshold)

class AnnotationManager:
    """
    Class for manipulation of JSON schema for COCO 1.0 annotations and semantic maps.
    Handles transformation from COCO 1.0 to our custom semantic map schema,
    as well as spatial queries and dynamic memory operations.
    """

    def __init__(self, json_file: str, map_file: str = "", schema_file: str = ""):
        self.json_file = json_file
        self.map_file = map_file
        self.schema_file = schema_file
        
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"JSON annotation file not found: {self.json_file}")
            
        try:
            self.data = load_json(self.json_file)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format (Error {e})")

        # Determine if it is already a Semantic Map or needs COCO transformation
        if 'map_info' in self.data:
            # Validate against Semantic Map Schema if schema file is provided
            if schema_file and not validate_json_schema(schema_file, self.data):
                raise ValueError("Invalid JSON data according to Semantic Map schema.")
            # Load stored embeddings to cache and convert strings to raw text for logic
            self._load_embeddings_to_cache()
            self._convert_json_to_raw_strings()
            self.map_spatial = MapSpatial(self.data)
        else:
            # It's a COCO 1.0 format map, transform it
            # (Note: we optionally validate it against the COCO schema if provided)
            if schema_file and not validate_json_schema(schema_file, self.data):
                raise ValueError("Invalid JSON data according to COCO schema.")
                
            self._transform_coco_to_semantic_map()

    def _transform_coco_to_semantic_map(self):
        """Initial transformation from COCO 1.0 to Semantic Map schema."""
        # Info extraction
        info = self.data.get('info', {})
        self.contributor = info.get('contributor', 'Unknown')
        self.date_created = info.get('date_created', 'Unknown')
        self.description = info.get('description', 'Transformed COCO 1.0 Map')
        self.version = info.get('version', '1.0')
        
        images = self.data.get('images', [])
        if images:
            self.width = images[0].get('width', 1000)
            self.height = images[0].get('height', 1000)
        else:
            self.width = 1000
            self.height = 1000
            
        self.map_url = f"file://{self.map_file}" if self.map_file else ""
        self.json_url = f"file://{self.json_file}"
        
        # Schema conversion steps
        self._delete_license()
        self._categories_to_regions()
        self._annotations_to_polygons()
        self._attributes_to_strings()
        self._images_to_map()
        self._supercategory_to_superregions()
        self._add_subregions()
        
        # Merge highly overlapping polygons (IoU >= 0.9)
        self.merge_polygons(sem_thresh=1.0, iou_threshold=0.9)
        
        self._reindex_regions()
        self._reindex_polygons()
        self._add_polygon_ids_to_regions()
        
        # Spatial initialization
        self.map_spatial = MapSpatial(self.data)
        self._compute_polygon_centroids()
        
        # Floating up common strings initially
        self.base_strings_to_regions(sem_thresh=1.0)
        
        # Populate map_info fields
        self.set_contributor(self.contributor)
        self.set_date_created(self.date_created)
        self.set_description(self.description)
        self.set_map_url(self.map_url)
        self.set_json_url(self.json_url)
        self.set_version(self.version)
        self.set_width(self.width)
        self.set_height(self.height)

    ####################################################################
    ##### Private JSON manipulation methods for map initialization #####
    ####################################################################

    def _delete_license(self):
        if 'licenses' in self.data:
            del self.data['licenses']

    def _categories_to_regions(self):
        self.data['regions'] = self.data.get('categories', [])
        if 'categories' in self.data:
            del self.data['categories']
            
        for ann in self.data.get('annotations', []):
            ann['region_ids'] = [ann['category_id']]
            if 'category_id' in ann:
                del ann['category_id']

    def _annotations_to_polygons(self):
        self.data['polygons'] = self.data.get('annotations', [])
        if 'annotations' in self.data:
            del self.data['annotations']
            
        for pol in self.data['polygons']:
            pol['polygon'] = pol.get('segmentation', [[]])[0]
            if 'segmentation' in pol:
                del pol['segmentation']
            if 'occluded' in pol.get('attributes', {}):
                del pol['attributes']['occluded']
            if 'iscrowd' in pol:
                del pol['iscrowd']
            if 'image_id' in pol:
                del pol['image_id']

    def _attributes_to_strings(self):
        for pol in self.data['polygons']:
            pol['images'] = []
            pol['strings'] = []
            if 'attributes' in pol:
                for key in pol['attributes']:
                    val = pol['attributes'][key]
                    if isinstance(val, str) and val.strip():
                        pol['strings'].append(val)
                del pol['attributes']

    def _images_to_map(self):
        self.data['map_info'] = {}
        if 'info' in self.data:
            del self.data['info']
        if 'images' in self.data:
            del self.data['images']

    def _supercategory_to_superregions(self):
        name_to_id = {reg['name']: reg['id'] for reg in self.data['regions']}
        for reg in self.data['regions']:
            supercat = reg.get('supercategory')
            if supercat and supercat in name_to_id:
                reg['superregions'] = [name_to_id[supercat]]
            else:
                reg['superregions'] = []
            if 'supercategory' in reg:
                del reg['supercategory']
            reg['strings'] = []
            reg['subregions'] = []

    def _add_subregions(self):
        for reg in self.data['regions']:
            for super_id in reg['superregions']:
                superreg = next((r for r in self.data['regions'] if r['id'] == super_id), None)
                if superreg:
                    if 'subregions' not in superreg:
                        superreg['subregions'] = []
                    superreg['subregions'].append(reg['id'])

    def _reindex_regions(self):
        old_to_new = {}
        for idx, reg in enumerate(self.data['regions']):
            old_to_new[reg['id']] = idx
            reg['id'] = idx
            
        for reg in self.data['regions']:
            reg['superregions'] = [old_to_new[rid] for rid in reg['superregions'] if rid in old_to_new]
            reg['subregions'] = [old_to_new[rid] for rid in reg['subregions'] if rid in old_to_new]
            
        for pol in self.data['polygons']:
            pol['region_ids'] = [old_to_new[rid] for rid in pol['region_ids'] if rid in old_to_new]

    def _reindex_polygons(self):
        old_to_new = {}
        for idx, pol in enumerate(self.data['polygons']):
            old_to_new[pol['id']] = idx
            pol['id'] = idx
            
        for reg in self.data['regions']:
            reg['polygon_ids'] = [old_to_new[pid] for pid in reg.get('polygon_ids', []) if pid in old_to_new]

    def _add_polygon_ids_to_regions(self):
        for reg in self.data['regions']:
            reg['polygon_ids'] = []
            for pol in self.data['polygons']:
                if reg['id'] in pol['region_ids']:
                    reg['polygon_ids'].append(pol['id'])

    def _compute_polygon_centroids(self):
        for pol in self.data['polygons']:
            try:
                centroid = self.map_spatial.get_polygon_centroid(pol['id'])
                pol['polygon_centroid'] = [centroid.x, centroid.y]
            except Exception:
                pol['polygon_centroid'] = [0.0, 0.0]

    def _load_embeddings_to_cache(self):
        model = embedding_model()
        for reg in self.data.get('regions', []):
            for s_doc in reg.get('strings', []):
                if isinstance(s_doc, dict) and 'string' in s_doc and 'embedding' in s_doc:
                    model.cache[f"text:{s_doc['string']}"] = s_doc['embedding']
        for pol in self.data.get('polygons', []):
            for s_doc in pol.get('strings', []):
                if isinstance(s_doc, dict) and 'string' in s_doc and 'embedding' in s_doc:
                    model.cache[f"text:{s_doc['string']}"] = s_doc['embedding']
            for img_doc in pol.get('images', []):
                if isinstance(img_doc, dict):
                    if 'description' in img_doc and 'description_embedding' in img_doc:
                        model.cache[f"text:{img_doc['description']}"] = img_doc['description_embedding']
        model._save_cache()

    def _convert_json_to_raw_strings(self):
        for reg in self.data.get('regions', []):
            reg['strings'] = [s['string'] for s in reg.get('strings', []) if isinstance(s, dict) and 'string' in s]
        for pol in self.data.get('polygons', []):
            pol['strings'] = [s['string'] for s in pol.get('strings', []) if isinstance(s, dict) and 'string' in s]

    def _get_serialized_data(self) -> dict:
        model = embedding_model()
        serialized = copy.deepcopy(self.data)
        
        for reg in serialized.get('regions', []):
            reg['strings'] = [{
                "string": s,
                "embedding": model.embed_text(s)
            } for s in reg.get('strings', [])]
            
        for pol in serialized.get('polygons', []):
            pol['strings'] = [{
                "string": s,
                "embedding": model.embed_text(s)
            } for s in pol.get('strings', [])]
            
            # Serialize images correctly
            serialized_images = []
            for img in pol.get('images', []):
                if isinstance(img, dict):
                    serialized_images.append(img)
                elif isinstance(img, str):
                    # It's a URL, let's create a placeholder ImageDoc
                    desc = f"Image at URL {img}"
                    serialized_images.append({
                        "image_url": img,
                        "description": desc,
                        "description_embedding": model.embed_text(desc),
                        "width": 640,
                        "height": 480,
                        "embedding": model.embed_text(desc)
                    })
            pol['images'] = serialized_images
        return serialized

    ###################
    ##### Getters #####
    ###################

    def get_contributor(self):
        return self.data['map_info']['contributor']

    def get_date_created(self):
        return self.data['map_info']['date_created']

    def get_description(self):
        return self.data['map_info']['description']

    def get_map_url(self):
        return self.data['map_info']['map_url']

    def get_json_url(self):
        return self.data['map_info']['json_url']

    def get_version(self):
        return self.data['map_info']['version']

    def get_width(self):
        return self.data['map_info']['width']

    def get_height(self):
        return self.data['map_info']['height']

    def get_polygons_region_name(self, region_name: str):
        region = next((r for r in self.data['regions'] if r['name'] == region_name), None)
        if not region:
            raise NameError(f"Region '{region_name}' not found")
        return region['polygon_ids']

    def get_polygons_region_id(self, reg_id: int):
        region = next((r for r in self.data['regions'] if r['id'] == reg_id), None)
        if not region:
            raise ValueError(f"Region ID {reg_id} not found")
        return region['polygon_ids']

    def get_regions_polygons(self, poly_list: List[int]) -> List[int]:
        region_list = []
        for poly_id in poly_list:
            polygon = next((p for p in self.data['polygons'] if p['id'] == poly_id), None)
            if polygon:
                region_list.extend(polygon['region_ids'])
        return list(set(region_list))
    
    def get_polygons_point(self, x_ind: float, y_ind: float) -> List[int]:
        return self.map_spatial.get_polygons_point(x_ind, y_ind)

    def get_regions_point(self, x_ind: float, y_ind: float) -> List[int]:
        return self.map_spatial.get_regions_point(x_ind, y_ind)

    ###################
    ##### Setters #####
    ###################

    def set_contributor(self, contributor: str):
        self.data['map_info']['contributor'] = contributor

    def set_date_created(self, date_created: str):
        self.data['map_info']['date_created'] = date_created

    def set_description(self, description: str):
        self.data['map_info']['description'] = description

    def set_map_url(self, map_url: str):
        self.data['map_info']['map_url'] = map_url

    def set_json_url(self, json_url: str):
        self.data['map_info']['json_url'] = json_url

    def set_version(self, version: str):
        self.data['map_info']['version'] = version

    def set_width(self, width: int):
        self.data['map_info']['width'] = int(width)

    def set_height(self, height: int):
        self.data['map_info']['height'] = int(height)

    def set_subregion_superregion_name(self, subregion_name: str, superregion_name: str):
        sub = next((r for r in self.data['regions'] if r['name'] == subregion_name), None)
        super_reg = next((r for r in self.data['regions'] if r['name'] == superregion_name), None)
        if sub and super_reg:
            self.set_subregion_superregion_id(sub['id'], super_reg['id'])

    def set_subregion_superregion_id(self, region_id: int, superregion_id: int):
        reg = next((r for r in self.data['regions'] if r['id'] == region_id), None)
        super_reg = next((r for r in self.data['regions'] if r['id'] == superregion_id), None)
        if reg and super_reg:
            if superregion_id not in reg['superregions']:
                reg['superregions'].append(superregion_id)
            if region_id not in super_reg['subregions']:
                super_reg['subregions'].append(region_id)

    def set_subregion_id(self, region_id: int, subregion_id: int):
        self.set_subregion_superregion_id(subregion_id, region_id)

    ###################################
    ##### JSON Read/write methods #####
    ###################################

    def read_json(self, file_path) -> dict:
        return load_json(file_path)

    def write_json(self, file_path):
        serialized_data = self._get_serialized_data()
        save_json(file_path, serialized_data, indent=2)

    def print_json(self):
        serialized_data = self._get_serialized_data()
        print(json.dumps(serialized_data, indent=2))

    ############################################
    ##### Public json manipulation methods #####
    ############################################

    def base_strings_to_regions(self, sem_thresh: float = 0.7):
        """Move strings common to all polygons of a region to the region level."""
        self.float_up_strings(sem_thresh=sem_thresh)

    def merge_polygons(self, sem_thresh: float = 0.7, iou_threshold: float = 0.9):
        """Merges polygons that overlap above the IoU threshold."""
        merged_polygons = []
        unmerged_polygons = copy.deepcopy(self.data['polygons'])

        while unmerged_polygons:
            poly = unmerged_polygons.pop(0)
            coords = poly['polygon']
            current_polygon_coords = [(x, y) for x, y in zip(coords[::2], coords[1::2])]
            if len(current_polygon_coords) < 3:
                merged_polygons.append(poly)
                continue
                
            merged = False
            for i, other in enumerate(unmerged_polygons):
                other_coords = other['polygon']
                other_polygon_coords = [(x, y) for x, y in zip(other_coords[::2], other_coords[1::2])]
                if len(other_polygon_coords) < 3:
                    continue
                    
                iou = polygon_iou(current_polygon_coords, other_polygon_coords)
                if iou >= iou_threshold:
                    # Merge strings and region_ids
                    poly['strings'] = list(set_union(set(poly['strings']), set(other['strings']), similarity_threshold=sem_thresh))
                    poly['region_ids'] = list(set(poly['region_ids'] + other['region_ids']))
                    # Combine image lists
                    poly['images'] = poly.get('images', []) + other.get('images', [])
                    
                    del unmerged_polygons[i]
                    merged_polygons.append(poly)
                    merged = True
                    break

            if not merged:
                merged_polygons.append(poly)

        self.data['polygons'] = merged_polygons

    def add_string(self, value: str, string_list: List[str], sem_thresh: float = 0.7) -> List[str]:
        """Adds a string value to a list of strings if it is semantically unique."""
        return list(set_union(set(string_list), {value}, similarity_threshold=sem_thresh))

    def subtract_string(self, value: str, string_list: List[str], sem_thresh: float = 0.7) -> List[str]:
        """Removes a string value from a list of strings if it is semantically equal."""
        return list(set_difference(set(string_list), {value}, similarity_threshold=sem_thresh))

    def add_string_polygon(self, poly_id: int, value: str, sem_thresh: float = 0.7):
        polygon = next((p for p in self.data['polygons'] if p['id'] == poly_id), None)
        if polygon:
            polygon['strings'] = self.add_string(value, polygon['strings'], sem_thresh=sem_thresh)

    def add_string_region_id(self, reg_id: int, value: str, sem_thresh: float = 0.7):
        region = next((r for r in self.data['regions'] if r['id'] == reg_id), None)
        if region:
            region['strings'] = self.add_string(value, region['strings'], sem_thresh=sem_thresh)

    def add_string_region_name(self, region_name: str, value: str, sem_thresh: float = 0.7):
        region = next((r for r in self.data['regions'] if r['name'] == region_name), None)
        if region:
            region['strings'] = self.add_string(value, region['strings'], sem_thresh=sem_thresh)

    def delete_string_polygon(self, poly_id: int, value: str, sem_thresh: float = 0.7):
        polygon = next((p for p in self.data['polygons'] if p['id'] == poly_id), None)
        if polygon:
            polygon['strings'] = self.subtract_string(value, polygon['strings'], sem_thresh=sem_thresh)

    def delete_string_region_id(self, reg_id: int, value: str, sem_thresh: float = 0.7):
        region = next((r for r in self.data['regions'] if r['id'] == reg_id), None)
        if region:
            region['strings'] = self.subtract_string(value, region['strings'], sem_thresh=sem_thresh)

    def delete_string_region_name(self, region_name: str, value: str, sem_thresh: float = 0.7):
        region = next((r for r in self.data['regions'] if r['name'] == region_name), None)
        if region:
            region['strings'] = self.subtract_string(value, region['strings'], sem_thresh=sem_thresh)

    def add_string_polygons_point(self, x_ind: float, y_ind: float, value: str, sem_thresh: float = 0.7):
        poly_list = self.get_polygons_point(x_ind, y_ind)
        for poly_id in poly_list:
            self.add_string_polygon(poly_id, value, sem_thresh=sem_thresh)

    def add_string_regions_point(self, x_ind: float, y_ind: float, value: str, sem_thresh: float = 0.7):
        region_list = self.get_regions_point(x_ind, y_ind)
        for reg_id in region_list:
            self.add_string_region_id(reg_id, value, sem_thresh=sem_thresh)

    def delete_string_polygons_region_id(self, reg_id: int, value: str, sem_thresh: float = 0.7):
        for pol in self.data['polygons']:
            if reg_id in pol.get('region_ids', []):
                pol['strings'] = self.subtract_string(value, pol['strings'], sem_thresh=sem_thresh)

    def delete_string_polygons_region_name(self, region_name: str, value: str, sem_thresh: float = 0.7):
        region = next((r for r in self.data['regions'] if r['name'] == region_name), None)
        if region:
            self.delete_string_polygons_region_id(region['id'], value, sem_thresh=sem_thresh)

    def delete_string_polygons_point(self, x_ind: float, y_ind: float, value: str, sem_thresh: float = 0.7):
        poly_list = self.get_polygons_point(x_ind, y_ind)
        for poly_id in poly_list:
            self.delete_string_polygon(poly_id, value, sem_thresh=sem_thresh)

    def delete_string_regions_point(self, x_ind: float, y_ind: float, value: str, sem_thresh: float = 0.7):
        region_list = self.get_regions_point(x_ind, y_ind)
        for reg_id in region_list:
            self.delete_string_region_id(reg_id, value, sem_thresh=sem_thresh)

    def add_image_polygon(self, poly_id: int, image_url: str, description: str = "", width: int = 640, height: int = 480):
        polygon = next((p for p in self.data['polygons'] if p['id'] == poly_id), None)
        if polygon:
            if not description:
                description = f"Image observation in polygon {poly_id}"
            model = embedding_model()
            # Fetch embeddings
            desc_embedding = model.embed_text(description)
            img_doc = {
                "image_url": image_url,
                "description": description,
                "description_embedding": desc_embedding,
                "width": width,
                "height": height,
                "embedding": desc_embedding  # Fallback to description embedding
            }
            polygon['images'].append(img_doc)

    ######################################################
    ##### Dynamic Memory Spatio-Semantic Operations #####
    ######################################################

    def float_up_strings(self, sem_thresh: float = 0.7):
        """
        Floats strings common to all polygons of a region up to the region level.
        Removes floated strings from the individual polygons to prevent redundancy.
        """
        for reg in self.data['regions']:
            reg_id = reg['id']
            # Find all polygons in this region
            region_polygons = [p for p in self.data['polygons'] if reg_id in p.get('region_ids', [])]
            if not region_polygons:
                continue
                
            # Get list of string sets for each polygon
            poly_string_sets = [set(p['strings']) for p in region_polygons]
            
            # Find the intersection of all polygon strings in the region
            common_strings = set_intersection(*poly_string_sets, similarity_threshold=sem_thresh)
            if not common_strings:
                continue
                
            # Remove common strings from polygons
            for p in region_polygons:
                p['strings'] = list(set_difference(set(p['strings']), common_strings, similarity_threshold=sem_thresh))
                
            # Add common strings to region
            reg['strings'] = list(set_union(set(reg['strings']), common_strings, similarity_threshold=sem_thresh))

    def proliferate_subregions(self, sem_thresh: float = 0.7, split_threshold: float = 0.7):
        """
        Splits a region if a plural subset of its polygons (>= 2) shares a set of strings
        that diverges semantically from the parent region's strings by more than split_threshold.
        """
        # We iterate over a copy of the regions to allow modifying the list
        current_regions = list(self.data['regions'])
        for reg in current_regions:
            reg_id = reg['id']
            # Find polygons in this region
            region_polygons = [p for p in self.data['polygons'] if reg_id in p.get('region_ids', [])]
            if len(region_polygons) < 2:
                continue
                
            # Collect all unique strings across all polygons in the region
            all_poly_strings = set()
            for p in region_polygons:
                all_poly_strings.update(p['strings'])
                
            # Find sets of polygons that share particular strings
            for s in all_poly_strings:
                # Find all polygons in the region containing this string (semantically equal)
                model = embedding_model()
                emb_s = model.embed_text(s)
                
                subset_polys = []
                for p in region_polygons:
                    for ps in p['strings']:
                        emb_ps = model.embed_text(ps)
                        # cos_similarity is in set_operations but we can just use set_difference / logic
                        from set_operations import cos_similarity
                        if cos_similarity(emb_s, emb_ps) >= sem_thresh:
                            subset_polys.append(p)
                            break
                            
                # Check if it is a plural subset but not all polygons (plural subset of size >= 2)
                if 2 <= len(subset_polys) < len(region_polygons):
                    # Compute common strings specifically shared by this subset of polygons
                    subset_string_sets = [set(p['strings']) for p in subset_polys]
                    shared_strings = set_intersection(*subset_string_sets, similarity_threshold=sem_thresh)
                    if not shared_strings:
                        continue
                        
                    # Calculate divergence from parent region strings
                    parent_strings = set(reg['strings'])
                    union_strings = set_union(parent_strings, shared_strings, similarity_threshold=sem_thresh)
                    
                    divergence = semantic_jaccard_distance(parent_strings, union_strings, similarity_threshold=sem_thresh)
                    if divergence >= split_threshold:
                        # Create new subregion!
                        new_id = len(self.data['regions'])
                        new_name = f"{reg['name']}_subregion_{new_id}"
                        
                        new_reg = {
                            "id": new_id,
                            "name": new_name,
                            "polygon_ids": [p['id'] for p in subset_polys],
                            "superregions": [reg_id],
                            "subregions": [],
                            "strings": list(shared_strings)
                        }
                        self.data['regions'].append(new_reg)
                        reg['subregions'].append(new_id)
                        
                        # Update polygons: associate with subregion and remove the shared strings
                        for p in subset_polys:
                            if new_id not in p['region_ids']:
                                p['region_ids'].append(new_id)
                            # Remove the shared strings from these polygons
                            p['strings'] = list(set_difference(set(p['strings']), shared_strings, similarity_threshold=sem_thresh))
                        
                        # Break and run reindex/recurse later or continue
                        break

    def merge_regions(self, sem_thresh: float = 0.7, merge_threshold: float = 0.7):
        """
        Merges two regions if their sets of strings are semantically similar (Jaccard similarity >= merge_threshold).
        """
        merged_any = False
        num_regions = len(self.data['regions'])
        
        for i in range(num_regions):
            for j in range(i + 1, num_regions):
                r1 = self.data['regions'][i]
                r2 = self.data['regions'][j]
                
                # Compute Semantic Jaccard Similarity between their string sets
                sim = semantic_jaccard_similarity(set(r1['strings']), set(r2['strings']), similarity_threshold=sem_thresh)
                if sim >= merge_threshold:
                    # Merge r2 into r1
                    r1['strings'] = list(set_union(set(r1['strings']), set(r2['strings']), similarity_threshold=sem_thresh))
                    r1['polygon_ids'] = list(set(r1['polygon_ids'] + r2['polygon_ids']))
                    r1['subregions'] = list(set(r1['subregions'] + r2['subregions']))
                    
                    # Update polygons originally pointing to r2
                    for pol in self.data['polygons']:
                        if r2['id'] in pol.get('region_ids', []):
                            pol['region_ids'] = [r1['id'] if rid == r2['id'] else rid for rid in pol['region_ids']]
                            pol['region_ids'] = list(set(pol['region_ids'])) # Deduplicate
                            
                    # Update superregions/subregions references in other regions
                    for reg in self.data['regions']:
                        reg['superregions'] = [r1['id'] if rid == r2['id'] else rid for rid in reg['superregions']]
                        reg['superregions'] = list(set(reg['superregions']))
                        reg['subregions'] = [r1['id'] if rid == r2['id'] else rid for rid in reg['subregions']]
                        reg['subregions'] = list(set(reg['subregions']))
                        
                    # Remove self-references
                    if r1['id'] in r1['superregions']:
                        r1['superregions'].remove(r1['id'])
                    if r1['id'] in r1['subregions']:
                        r1['subregions'].remove(r1['id'])
                        
                    # Remove r2 from the list
                    self.data['regions'].remove(r2)
                    merged_any = True
                    break
            if merged_any:
                break
                
        if merged_any:
            # Reindex to keep indices contiguous and valid
            self._reindex_regions()
            self._reindex_polygons()
            self._add_polygon_ids_to_regions()
            # Recurse to find other merges
            self.merge_regions(sem_thresh=sem_thresh, merge_threshold=merge_threshold)

    def add_observation(
        self, 
        x: float, 
        y: float, 
        strings: List[str], 
        image_url: str = None, 
        sem_thresh: float = 0.7, 
        split_thresh: float = 0.7, 
        merge_thresh: float = 0.7
    ):
        """
        Adds a semantic observation (strings/images) at coordinates (x, y).
        Triggers upward floating, subregion splitting, and region merging,
        then writes the updated memory store back to disk.
        """
        poly_list = self.get_polygons_point(x, y)
        
        # If no polygon covers the point, create a small 10x10 square polygon centered at (x, y)
        if not poly_list:
            new_poly_id = len(self.data['polygons'])
            # Create square
            coords = [
                x - 5.0, y - 5.0,
                x + 5.0, y - 5.0,
                x + 5.0, y + 5.0,
                x - 5.0, y + 5.0
            ]
            
            # Find or create a default "Observation Zone" region
            obs_region = next((r for r in self.data['regions'] if r['name'] == 'Observation Zone'), None)
            if not obs_region:
                obs_id = len(self.data['regions'])
                obs_region = {
                    "id": obs_id,
                    "name": "Observation Zone",
                    "polygon_ids": [new_poly_id],
                    "superregions": [],
                    "subregions": [],
                    "strings": []
                }
                self.data['regions'].append(obs_region)
            else:
                obs_region['polygon_ids'].append(new_poly_id)
                
            new_poly = {
                "id": new_poly_id,
                "region_ids": [obs_region['id']],
                "polygon": coords,
                "polygon_centroid": [x, y],
                "strings": [],
                "images": []
            }
            self.data['polygons'].append(new_poly)
            poly_list = [new_poly_id]
            
            # Reinitialize map spatial to recognize new polygon
            self.map_spatial = MapSpatial(self.data)

        # Append strings and image to target polygons
        for poly_id in poly_list:
            for val in strings:
                self.add_string_polygon(poly_id, val, sem_thresh=sem_thresh)
            if image_url:
                self.add_image_polygon(poly_id, image_url)
                
        # Trigger Dynamic Memory Operations
        self.float_up_strings(sem_thresh=sem_thresh)
        self.proliferate_subregions(sem_thresh=sem_thresh, split_threshold=split_thresh)
        self.merge_regions(sem_thresh=sem_thresh, merge_threshold=merge_thresh)
        
        # Reinitialize spatial representation
        self.map_spatial = MapSpatial(self.data)
        self._compute_polygon_centroids()
        
        # Save updated memory store
        self.write_json(self.json_file)
