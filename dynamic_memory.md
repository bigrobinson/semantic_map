# **Dynamic Memory Requirements**

- The base map is a 2D grid of elements (pixels) representing a part of the world.  
- The map elements contain cost or occupancy values to represent the presence of stuff.   
- A polygon (a cyclic list of map coordinates) describes a simply connected area within the map.  
- A region of the map is, in general, a list of polygons—it may be simply connected (have only one polygon) or not.  
- A region attains semantic meaning through its association with a list of string expressions.  
- The set of regions, subregions, and polygons and their associated meanings (string expressions) constitute the semantic map layer.  
- Spatial coordinates are pixel indices in the base map and spatial meaning derives from the spatial relationships among this regular grid of pixels, which are represented by 2D vectors with integer components. Spatial similarity is represented by Euclidean distance.  
- Semantic coordinates are vector embeddings of the string expressions associated with regions. Semantic meaning derives from the semantic relationships among the strings, which are represented by high-dimensional embedding vectors. Semantic similarity is represented by the similarity metric.  
- Spatial relationships are encoded in the base map. Semantic relationships are encoded in the semantic map. There is no need to capture relationships among objects (pixels, regions) by giving state to the objects themselves. It’s all implied in the maps via coordinates and similarity metrics.  
- The distance metric for the base map is Euclidean distance. The distance metric for the semantic map is similarity metric. Space and meaning are connected via regions, which exhibit Euclidean proximity and semantic similarity to one another.  
- In the following, we overload the equality operator with the notion of semantic equality. Semantic equality is defined in this way: If the similarity metric between the embeddings of two statements is higher than a predetermined threshold, then the two statements are “equal”. Semantic set operations such as union, intersection, difference, and so on are also overloaded, due to the overloaded equality operation. The same applies to the notions of uniqueness.  
- A region must be associated with a unique *set* of string expressions (meanings) in the sense that the similarity metric of the embedding of the string expressions with all other regions is below a threshold. No other region may have the same set of strings (i.e. the same meaning).  
- A plural subset of the polygons forming a region may acquire new strings in common. If the union of this new set of strings with the region’s original strings diverge semantically, by more than a threshold, from the set of original strings, then that subset of polygons becomes a new subregion. Then there is one more region.  
- If one region, through acquisition or loss of string expressions, comes to possess a similar set of strings as another region, then the one region becomes a part of the other, and there is one less region.  
- A human or a competent robot may add strings to polygons or even add polygons based on observations.  
- Observations are made at points in space and their meanings, in the form of strings and images, are appended to the polygons that cover those points. As time goes on, the polygons become more specialized with local knowledge while common knowledge floats upward in the hierarchy. This will naturally lead to the proliferation of regions and subregions according to the simple dynamics described above.  
  