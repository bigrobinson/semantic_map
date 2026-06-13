from __future__ import annotations # fix "NameError: name 'StringDoc' is not defined"
import os
from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl, ID, AnyTensor
from pydantic import Field
from typing import List

"""
DocArray schemas for a map.
"""

class MapInfoDoc(BaseDoc):
    """
    Some basic information about the map.
    """
    id:  ID = None
    contributor: str = Field(description = "the name of the person who made the map")
    date_created: str = Field(description = "the date the map was created")
    description: str = Field(description = "a description of the map")
    map_url: str = Field(description = "the url of the original annotation JSON from CVAT")
    json_url: str = Field(description = "the url of the original annotation JSON from CVAT")
    version: str = Field(description = "the version number of the map")
    width: int = Field(description = "the width of the map in pixels")
    height: int = Field(description = "the height of the map in pixels")

class StringDoc(BaseDoc):
    """
    A string that tells us things about the polygons or regions of the map within which it lies.
    """
    string: str = Field(description = "a string that tells us something about the containing polygon or region")
    embedding: AnyTensor = Field(space = "cosine_sim", dim=768, description = "the vector embedding of the string")

class ImageDoc(BaseDoc):
    """
    An image that was acquired within a specific polygon on the map.
    """
    image_url: ImageUrl = Field(description = "the url of an image that was acquired within the polygon")
    description: str = Field(description = "a textual description of the image")
    description_embedding: AnyTensor = Field(space = "cosine_sim", dim=768, description = "the vector embedding of the text description")
    width: int = Field(description = "the width of the image in pixels")
    height: int = Field(description = "the height of the image in pixels")
    embedding: AnyTensor = Field(space = "cosine_sim", dim=768, description = "the vector embedding of the image")

class PolygonDoc(BaseDoc):
    """
    A basic polygon that describes a place of a certain shape and location within the map.
    """
    id: int = Field(description = "the id and the direct index of the polygon")
    region_ids: List[int] = Field(description = "the id's of the regions that contain the polygon")
    polygon: List[float] = Field(description = "the coordinates of the vertices of the polygon")
    polygon_centroid: AnyTensor = Field(space = "sqeuclidean_dist", dim=2, description = "the coordinates of the centroid of the polygon")
    strings: DocList[StringDoc] = Field(description = "the strings that tell us something about the polygon")
    images: DocList[ImageDoc] = Field(description = "the url's of the images that the polygon contains")

class RegionDoc(BaseDoc):
    """
    A region consisting of a set of polygons that all have some common meaning or meanings. It can contain or be contained by other regions.
    """
    id: int = Field(description = "the id and the direct index of the region")
    name: str = Field(description = "the descriptive name of the region")
    polygon_ids: List[int] = Field(description = "id's of polygons that this region wholly contains")
    superregions: List[int] = Field(description = "the superregions that wholly contain this region")
    subregions: List[int] = Field(description = "the subregions that this region wholly contains")
    strings: DocList[StringDoc] = Field(description = "the strings that tell us something about the region")

