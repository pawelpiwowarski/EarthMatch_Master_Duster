import argparse
import torch
import logging
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from pyproj import Geod

sys.path.append(str(Path('image-matching-models')))
sys.path.append(str(Path('image-matching-models/third_party/RoMa')))
sys.path.append(str(Path('image-matching-models/third_party/duster')))
sys.path.append(str(Path('image-matching-models/third_party/DeDoDe')))
sys.path.append(str(Path('image-matching-models/third_party/Steerers')))
sys.path.append(str(Path('image-matching-models/third_party/Se2_LoFTR')))
sys.path.append(str(Path('image-matching-models/third_party/LightGlue')))
sys.path.append(str(Path('image-matching-models/third_party/imatch-toolbox')))

import util_matching

# Initialize geodesic calculator for WGS84 ellipsoid
geod = Geod(ellps="WGS84")

def load_torch_file(file_path):
    """Load a .torch file using PyTorch."""
    try:
        return torch.load(file_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {str(e)}")

def compute_threshold(all_results):
    """Calculate the optimal inlier threshold using true/false positives."""
    tp_inliers = [res[2] for res in all_results if res[4]]
    fp_inliers = [res[2] for res in all_results if not res[4]]
    return -1 if not fp_inliers else util_matching.compute_threshold(tp_inliers, fp_inliers, 0.999)

def compute_evaluation_metrics(all_results, threshold, data_dir):
    """Compute metrics with geodesic area and distance calculations."""
    results_dict = defaultdict(list)
    for res in all_results:
        results_dict[res[0]].append(res)

    query_folders = sorted(list(Path(data_dir).glob("*")))
    total_queries = len(query_folders)
    
    metrics = {
        "total": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "fclt_le_200": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "fclt_200_400": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "fclt_400_800": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "fclt_g_800": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "tilt_l_40": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "tilt_ge_40": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "cldp_l_40": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "cldp_ge_40": {"sum_distance": 0.0, "sum_area": 0.0, "count": 0},
        "_meta": {
            "total_queries": total_queries,
            "failed_matches": 0,
            "processed_queries": 0,
            "threshold": threshold
        }
    }

    for query_folder in tqdm(query_folders, desc="Processing queries"):
        paths = sorted(list(query_folder.glob("*")))
        if not paths:
            continue
            
        query_path = paths[0]
        query_id = query_path.stem
        query_results = results_dict.get(query_id, [])

        try:
            gt_center = util_matching.get_centerpoint_from_query_path(query_path)
            gt_lat, gt_lon = gt_center.x, gt_center.y
            
            # Filter candidates by threshold
            filtered_results = [
                res for res in query_results 
                if (threshold == -1) or (res[2] >= threshold)
            ]
            
            if not filtered_results:
                metrics["_meta"]["failed_matches"] += 1
                continue

            distances, areas = [], []
            for res in filtered_results:
                footprint_tensor = res[3]
                footprint_np = footprint_tensor.numpy()
                polygon = util_matching.get_polygon(footprint_np)

                if not polygon.is_valid:
                    raise ValueError("Invalid polygon")
                
                # Calculate geodesic area
                coords = np.array(polygon.exterior.coords)
                lat, lon = coords[:, 0], coords[:, 1]

                area_m2, _ = geod.polygon_area_perimeter(lon, lat)
                area_km2 = abs(area_m2) / 1e6  # Convert to km²
                
                # Calculate geodesic distance
                centroid = polygon.centroid
                clat, clon = centroid.x, centroid.y
                _, _, distance_m = geod.inv(clon, clat, gt_lon, gt_lat)
                distance_km = distance_m / 1000
                
                # Add validation checks
                if np.isnan(distance_km) or np.isnan(area_km2):
                    raise(Exception('distance is nan'))
                if area_km2 <= 0:
                    raise(Exception('area in square km is less than 0'))
                    
                distances.append(distance_km)
                areas.append(area_km2)

            if not distances:
                metrics["_meta"]["failed_matches"] += 1
                continue

            # Find candidate with minimum distance
            min_distance_idx = np.argmin(distances)
            best_distance = distances[min_distance_idx]
            best_area = areas[min_distance_idx]

            # Update metrics with best candidate
            metrics["total"]["sum_distance"] += best_distance
            metrics["total"]["sum_area"] += best_area
            metrics["total"]["count"] += 1
            metrics["_meta"]["processed_queries"] += 1

            # Update categories
            def update_category(condition, category):
                if condition(query_id):
                    metrics[category]["sum_distance"] += best_distance
                    metrics[category]["sum_area"] += best_area
                    metrics[category]["count"] += 1

            update_category(util_matching.fclt_le_200, "fclt_le_200")
            update_category(util_matching.fclt_200_400, "fclt_200_400")
            update_category(util_matching.fclt_400_800, "fclt_400_800")
            update_category(util_matching.fclt_g_800, "fclt_g_800")
            update_category(util_matching.tilt_l_40, "tilt_l_40")
            update_category(util_matching.tilt_ge_40, "tilt_ge_40")
            update_category(util_matching.cldp_l_40, "cldp_l_40")
            update_category(util_matching.cldp_ge_40, "cldp_ge_40")

        except Exception as e:
            logging.error(f"Error processing {query_id}: {str(e)}")
            continue

    return metrics

def display_metrics(metrics, console):
    """Display metrics with proper geographic units."""
    table = Table(
        title=f"[bold]Geographic Evaluation (Threshold: {metrics['_meta']['threshold']})[/bold]",
        show_header=True,
        header_style="bold magenta",
        box=None
    )
    table.add_column("Category", style="cyan", min_width=15)
    table.add_column("Queries", style="green", justify="right")
    table.add_column("Avg Distance →", style="magenta", justify="right")
    table.add_column("Avg Area →", style="cyan", justify="right")

    for key in metrics:
        if key.startswith("_"): continue
        
        data = metrics[key]
        count = data["count"]
        avg_distance = data["sum_distance"] / count if count > 0 else 0.0
        avg_area = data["sum_area"] / count if count > 0 else 0.0
        
        table.add_row(
            key.replace("_", " ").title(),
            f"{count}",
            f"{avg_distance:.2f} km",
            f"{avg_area:.2f} km²"
        )

    # Add meta statistics
    meta = metrics["_meta"]
    table.add_row(
        ":globe_with_meridians: [bold]Total Queries[/bold]",
        f"{meta['total_queries']}",
        style="bold"
    )
    table.add_row(
        ":white_check_mark: [bold]Processed[/bold]",
        f"{meta['processed_queries']}",
        style="bold green"
    )
    table.add_row(
        ":x: [bold]Failed[/bold]",
        f"{meta['failed_matches']}",
        style="bold red"
    )

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Evaluate geographic footprint matching")
    parser.add_argument("results_file", help="Path to results.torch file")
    parser.add_argument("--data_dir", type=str, default="./data", 
                      help="Directory containing query folders")
    args = parser.parse_args()

    console = Console()
    try:
        data = load_torch_file(args.results_file)
        threshold = compute_threshold(data)
        metrics = compute_evaluation_metrics(data, threshold, args.data_dir)
        
        console.print(f"\n[bold]Geographic Evaluation Summary[/bold]")
        console.print(f"- Automatic threshold: {threshold}")
        console.print(f"- Successful matches: {metrics['_meta']['processed_queries']}")
        console.print(f"- Failed matches: {metrics['_meta']['failed_matches']}\n")
        
        display_metrics(metrics, console)
        
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {str(e)}")

if __name__ == "__main__":
    main()