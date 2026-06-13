from __future__ import annotations
from shapely.geometry import Point, Polygon, MultiPoint
from typing import List, Tuple, Dict, Any

class MapSpatial:
    """
    A class to handle spatial analysis of the semantic map.
    """
    def __init__(self, map_data: dict):
        self.data = map_data
        if 'regions' not in self.data:
            self.data['regions'] = []
        if 'polygons' not in self.data:
            self.data['polygons'] = []

    def get_polygons_point(self, x_ind: float, y_ind: float) -> List[int]:
        """
        Get the IDs of polygons within which a point on the map lies.
        """
        polygon_list = []
        point = Point(x_ind, y_ind)
        for pol in self.data['polygons']:
            coords = pol['polygon']
            # Reconstruct the polygon vertices (x, y)
            poly_vertices = [(x, y) for x, y in zip(coords[::2], coords[1::2])]
            if len(poly_vertices) < 3:
                continue
            polygon = Polygon(poly_vertices)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if point.within(polygon):
                polygon_list.append(pol['id'])
        return polygon_list

    def get_regions_point(self, x_ind: float, y_ind: float) -> List[int]:
        """
        Get the IDs of regions within which a point on the map lies.
        """
        region_list = []
        polygon_list = self.get_polygons_point(x_ind, y_ind)
        for poly_id in polygon_list:
            # Find the polygon by ID
            polygon = next((p for p in self.data['polygons'] if p['id'] == poly_id), None)
            if polygon and 'region_ids' in polygon:
                region_list.extend(polygon['region_ids'])
        return list(set(region_list))

    def get_polygon_centroid(self, poly_id: int) -> Point:
        """
        Get the centroid of a polygon by its ID as a Shapely Point.
        """
        polygon = next((p for p in self.data['polygons'] if p['id'] == poly_id), None)
        if not polygon:
            raise ValueError(f"Polygon with ID {poly_id} not found.")
        coords = polygon['polygon']
        poly_vertices = [(x, y) for x, y in zip(coords[::2], coords[1::2])]
        if len(poly_vertices) < 3:
            # Fallback to average coordinate if not enough vertices for a Polygon
            x_coords = coords[::2]
            y_coords = coords[1::2]
            return Point(sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))
            
        shapely_poly = Polygon(poly_vertices)
        if not shapely_poly.is_valid:
            shapely_poly = shapely_poly.buffer(0)
        return shapely_poly.centroid

    def get_region_centroid_id(self, reg_id: int) -> Point:
        """
        Get the centroid of a region by its ID.
        """
        region = next((r for r in self.data['regions'] if r['id'] == reg_id), None)
        if not region:
            raise ValueError(f"Region with ID {reg_id} not found.")
            
        point_list = []
        for poly_id in region.get('polygon_ids', []):
            try:
                point_list.append(self.get_polygon_centroid(poly_id))
            except ValueError:
                continue
                
        if not point_list:
            raise ValueError(f"Region with ID {reg_id} contains no valid polygons.")
            
        return MultiPoint(point_list).centroid

    def get_region_centroid_name(self, region_name: str) -> Point:
        """
        Get the centroid of a region by its name.
        """
        region = next((r for r in self.data['regions'] if r['name'] == region_name), None)
        if not region:
            raise NameError(f"Region named '{region_name}' not found.")
        return self.get_region_centroid_id(region['id'])
