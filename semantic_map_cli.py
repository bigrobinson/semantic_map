#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np

# Add parent directory to sys.path to resolve relative imports if run as main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotation_manager import AnnotationManager
from set_operations import cos_similarity, get_embedding
from map_spatial import MapSpatial

def cmd_import_coco(args):
    """Transforms COCO 1.0 JSON to the Semantic Map JSON format."""
    print(f"Importing COCO annotations from {args.coco_json}...")
    try:
        manager = AnnotationManager(
            json_file=args.coco_json,
            map_file=args.map_image or "",
            schema_file=args.coco_schema or ""
        )
        # Apply initial floating up of common strings with args.sem_thresh
        manager.float_up_strings(sem_thresh=args.sem_thresh)
        
        # Write to target output file
        manager.write_json(args.output)
        print(f"Success! Semantic Map successfully created at: {args.output}")
    except Exception as e:
        print(f"Error importing COCO file: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_query_spatial(args):
    """Finds regions and polygons covering a coordinate (x, y)."""
    if not os.path.exists(args.map_json):
        print(f"Error: Map JSON file not found at {args.map_json}", file=sys.stderr)
        sys.exit(1)
        
    try:
        manager = AnnotationManager(json_file=args.map_json)
        polygons = manager.get_polygons_point(args.x, args.y)
        regions = manager.get_regions_point(args.x, args.y)
        
        results = {
            "query": {"x": args.x, "y": args.y},
            "polygons": [],
            "regions": []
        }
        
        for pid in polygons:
            poly = next((p for p in manager.data['polygons'] if p['id'] == pid), None)
            if poly:
                results["polygons"].append({
                    "id": pid,
                    "centroid": poly.get("polygon_centroid"),
                    "strings": poly.get("strings", [])
                })
                
        for rid in regions:
            reg = next((r for r in manager.data['regions'] if r['id'] == rid), None)
            if reg:
                # Find centroid using MapSpatial helper
                centroid = manager.map_spatial.get_region_centroid_id(rid)
                results["regions"].append({
                    "id": rid,
                    "name": reg.get("name"),
                    "centroid": [centroid.x, centroid.y],
                    "strings": reg.get("strings", [])
                })
                
        output_str = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"Success! Query results written to: {args.output}")
        else:
            print(output_str)
            
    except Exception as e:
        print(f"Error querying spatial location: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_query_semantic(args):
    """Searches for regions and polygons matching a text query."""
    if not os.path.exists(args.map_json):
        print(f"Error: Map JSON file not found at {args.map_json}", file=sys.stderr)
        sys.exit(1)
        
    try:
        manager = AnnotationManager(json_file=args.map_json)
        query_emb = get_embedding(args.query)
        
        matches = []
        
        # Search regions
        for reg in manager.data.get('regions', []):
            max_sim = -1.0
            best_string = ""
            for s in reg.get('strings', []):
                s_emb = get_embedding(s)
                sim = cos_similarity(query_emb, s_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_string = s
            if max_sim >= 0:
                centroid = manager.map_spatial.get_region_centroid_id(reg['id'])
                matches.append({
                    "type": "region",
                    "id": reg['id'],
                    "name": reg['name'],
                    "centroid": [centroid.x, centroid.y],
                    "matched_string": best_string,
                    "similarity": max_sim,
                    "all_strings": reg.get('strings', [])
                })
                
        # Search polygons
        for pol in manager.data.get('polygons', []):
            max_sim = -1.0
            best_string = ""
            for s in pol.get('strings', []):
                s_emb = get_embedding(s)
                sim = cos_similarity(query_emb, s_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_string = s
            if max_sim >= 0:
                matches.append({
                    "type": "polygon",
                    "id": pol['id'],
                    "centroid": pol.get('polygon_centroid', [0.0, 0.0]),
                    "matched_string": best_string,
                    "similarity": max_sim,
                    "all_strings": pol.get('strings', [])
                })
                
        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = matches[:args.k]
        
        results = {
            "query": args.query,
            "results": top_k
        }
        
        output_str = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"Success! Query results written to: {args.output}")
        else:
            print(output_str)
            
    except Exception as e:
        print(f"Error querying semantics: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_add_observation(args):
    """Adds observation attributes at a coordinate (x, y) and triggers updates."""
    if not os.path.exists(args.map_json):
        print(f"Error: Map JSON file not found at {args.map_json}", file=sys.stderr)
        sys.exit(1)
        
    try:
        manager = AnnotationManager(json_file=args.map_json)
        print(f"Adding observation strings {args.strings} at ({args.x}, {args.y})...")
        
        manager.add_observation(
            x=args.x,
            y=args.y,
            strings=args.strings,
            image_url=args.image_url,
            sem_thresh=args.sem_thresh,
            split_thresh=args.split_thresh,
            merge_thresh=args.merge_thresh
        )
        
        print("Dynamic memory updates complete (upward floating, subregion splitting, region merging).")
        
        # Optionally write summary of changes to output
        if args.output:
            # Re-read to confirm and output
            summary = {
                "status": "success",
                "coordinates": {"x": args.x, "y": args.y},
                "added_strings": args.strings,
                "image_url": args.image_url,
                "total_polygons": len(manager.data['polygons']),
                "total_regions": len(manager.data['regions'])
            }
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Success! Observation details saved to: {args.output}")
            
    except Exception as e:
        print(f"Error adding observation: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Metric-Semantic Memory System CLI for OpenClaw Robots")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")
    
    # import-coco subparser
    parser_import = subparsers.add_parser("import-coco", help="Import CVAT COCO annotations to Semantic Map JSON")
    parser_import.add_argument("--coco-json", required=True, help="Path to COCO JSON annotation file")
    parser_import.add_argument("--output", required=True, help="Path to save the output Semantic Map JSON")
    parser_import.add_argument("--map-image", help="Path/URL to the base occupancy grid/costmap image")
    parser_import.add_argument("--coco-schema", help="Path to COCO validation schema")
    parser_import.add_argument("--sem-thresh", type=float, default=0.7, help="Semantic similarity threshold")
    parser_import.set_defaults(func=cmd_import_coco)
    
    # query-spatial subparser
    parser_spatial = subparsers.add_parser("query-spatial", help="Query polygons and regions containing a coordinate (x, y)")
    parser_spatial.add_argument("--map-json", required=True, help="Path to Semantic Map JSON file")
    parser_spatial.add_argument("-x", type=float, required=True, help="X coordinate")
    parser_spatial.add_argument("-y", type=float, required=True, help="Y coordinate")
    parser_spatial.add_argument("--output", help="Path to save results JSON (otherwise printed to stdout)")
    parser_spatial.set_defaults(func=cmd_query_spatial)
    
    # query-semantic subparser
    parser_semantic = subparsers.add_parser("query-semantic", help="Perform semantic search on map strings")
    parser_semantic.add_argument("--map-json", required=True, help="Path to Semantic Map JSON file")
    parser_semantic.add_argument("-q", "--query", required=True, help="Search query string")
    parser_semantic.add_argument("-k", type=int, default=5, help="Number of results to return")
    parser_semantic.add_argument("--output", help="Path to save results JSON (otherwise printed to stdout)")
    parser_semantic.set_defaults(func=cmd_query_semantic)
    
    # add-observation subparser
    parser_obs = subparsers.add_parser("add-observation", help="Record an observation at (x, y) and update map memory")
    parser_obs.add_argument("--map-json", required=True, help="Path to Semantic Map JSON file")
    parser_obs.add_argument("-x", type=float, required=True, help="X coordinate")
    parser_obs.add_argument("-y", type=float, required=True, help="Y coordinate")
    parser_obs.add_argument("-s", "--strings", nargs="+", required=True, help="Semantic strings observed")
    parser_obs.add_argument("--image-url", help="URL of image observed")
    parser_obs.add_argument("--sem-thresh", type=float, default=0.7, help="Semantic similarity threshold")
    parser_obs.add_argument("--split-thresh", type=float, default=0.7, help="Subregion splitting threshold")
    parser_obs.add_argument("--merge-thresh", type=float, default=0.7, help="Region merging threshold")
    parser_obs.add_argument("--output", help="Path to save operation summary JSON")
    parser_obs.set_defaults(func=cmd_add_observation)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
