import numpy as np
from scipy.spatial.distance import cosine
from shapely.geometry import Polygon
import shapely.errors
from typing import List, Set, Any
import models

# Get embedding model singleton
EMBEDDING_MODEL = models.embedding_model()

def get_embedding(input_val: Any) -> np.ndarray:
    """Gets the embedding vector for a string, numpy array, or a dict containing an embedding."""
    if isinstance(input_val, dict) and "embedding" in input_val:
        return np.asarray(input_val["embedding"])
    elif isinstance(input_val, str):
        return np.asarray(EMBEDDING_MODEL.embed_text(input_val))
    elif isinstance(input_val, np.ndarray):
        return np.asarray(EMBEDDING_MODEL.embed_image(input_val))
    else:
        raise ValueError("Input must be a string, a numpy array, or a dict containing 'embedding'.")

def cos_similarity(vec1: Any, vec2: Any) -> float:
    """Calculates cosine similarity between two vectors."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Both input vectors must be 1-D.")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    # cosine() from scipy returns 1 - cos_sim
    return float(1.0 - cosine(vec1, vec2))

def polygon_iou(polygon1_coords: List[tuple], polygon2_coords: List[tuple]) -> float:
    """Calculates the Intersection over Union (IoU) between two polygons."""
    try:
        polygon1 = Polygon(polygon1_coords)
        polygon2 = Polygon(polygon2_coords)

        if not polygon1.is_valid:
            polygon1 = polygon1.buffer(0)
        if not polygon2.is_valid:
            polygon2 = polygon2.buffer(0)

        intersection_area = polygon1.intersection(polygon2).area
        union_area = polygon1.union(polygon2).area

        if union_area == 0:
            return 0.0

        return float(intersection_area / union_area)
    except Exception:
        return 0.0

# Set operations with equality overloaded as semantic similarity.
def set_union(*sets: Set[str], similarity_threshold: float = 0.7, eps: float = 1e-6) -> Set[str]:
    """Returns the union of multiple sets of strings based on cosine similarity."""
    union_set = set()
    for current_set in sets:
        for item in current_set:
            embedding2 = get_embedding(item)
            is_similar = False
            for existing_item in union_set:
                embedding1 = get_embedding(existing_item)
                if cos_similarity(embedding1, embedding2) >= similarity_threshold - eps:
                    is_similar = True
                    break
            if not is_similar:
                union_set.add(item)
    return union_set

def set_intersection(*sets: Set[str], similarity_threshold: float = 0.7, eps: float = 1e-6) -> Set[str]:
    """Returns the intersection of multiple sets of strings based on cosine similarity."""
    if len(sets) == 0:
        return set()
    elif len(sets) == 1:
        return sets[0]
        
    intersection_set = sets[0].copy()
    for current_set in sets[1:]:
        for item in list(intersection_set):
            embedding1 = get_embedding(item)
            found = False
            for item2 in current_set:
                embedding2 = get_embedding(item2)
                if cos_similarity(embedding1, embedding2) >= similarity_threshold - eps:
                    found = True
                    break
            if not found:
                intersection_set.remove(item)
    return intersection_set

def set_difference(set1: Set[str], set2: Set[str], similarity_threshold: float = 0.7, eps: float = 1e-6) -> Set[str]:
    """Returns the difference (set1 - set2) of sets of strings based on cosine similarity."""
    difference_set = set1.copy()
    for item in set2:
        embedding2 = get_embedding(item)
        for existing_item in list(difference_set):
            embedding1 = get_embedding(existing_item)
            similarity = cos_similarity(embedding1, embedding2)
            if similarity >= similarity_threshold - eps:
                difference_set.remove(existing_item)
    return difference_set