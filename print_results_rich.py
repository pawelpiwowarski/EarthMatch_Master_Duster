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
from shapely.geometry import Polygon, box

sys.path.append(str(Path('image-matching-models')))
sys.path.append(str(Path('image-matching-models/third_party/RoMa')))
sys.path.append(str(Path('image-matching-models/third_party/duster')))
sys.path.append(str(Path('image-matching-models/third_party/DeDoDe')))
sys.path.append(str(Path('image-matching-models/third_party/Steerers')))
sys.path.append(str(Path('image-matching-models/third_party/Se2_LoFTR')))
sys.path.append(str(Path('image-matching-models/third_party/LightGlue')))
sys.path.append(str(Path('image-matching-models/third_party/imatch-toolbox')))

import util_matching

def load_torch_file(file_path):
    """Load a .torch file using PyTorch."""
    try:
        return torch.load(file_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {str(e)}")

def compute_threshold(all_results):
    """Calculate the optimal inlier threshold."""
    tp_inliers = [res[2] for res in all_results if res[-1]]
    fp_inliers = [res[2] for res in all_results if not res[-1]]
    return -1 if not fp_inliers else util_matching.compute_threshold(tp_inliers, fp_inliers, 0.999)

def compute_evaluation_metrics(all_results, threshold, data_dir):
    """Compute metrics while properly accounting for all queries."""
    # Group results by query name
    results_dict = defaultdict(list)
    for res in all_results:
        results_dict[res[0]].append(res)

    # Get all query folders from data directory
    query_folders = sorted(list(Path(data_dir).glob("*")))
    total_queries = len(query_folders)
    
    metrics = {
        "total": {"correct": 0, "total": 0},
        "fclt_le_200": {"correct": 0, "total": 0},
        "fclt_200_400": {"correct": 0, "total": 0},
        "fclt_400_800": {"correct": 0, "total": 0},
        "fclt_g_800": {"correct": 0, "total": 0},
        "tilt_l_40": {"correct": 0, "total": 0},
        "tilt_ge_40": {"correct": 0, "total": 0},
        "cldp_l_40": {"correct": 0, "total": 0},
        "cldp_ge_40": {"correct": 0, "total": 0},
        "_meta": {
            "total_queries": total_queries,
            "no_valid_candidates": 0,
            "failed_matches": 0,
            "successful_matches": 0
        }
    }

    for query_folder in tqdm(query_folders, desc="Processing queries"):
        paths = sorted(list(query_folder.glob("*")))

        
        query_path = paths[0]

   

        
        try:
            # Get ground truth information
            gt_center = util_matching.get_centerpoint_from_query_path(query_path)

            has_valid_candidates = False
            
            # Check all candidate images in folder
            for candidate_path in query_folder.glob("*"):
                if candidate_path == query_path:
                    continue
                
                # Get candidate footprint from path
                candidate_footprint = util_matching.path_to_footprint(candidate_path)
                pred_polygon = util_matching.get_polygon(candidate_footprint.numpy())

                
                if pred_polygon.contains(gt_center):
                    has_valid_candidates = True
                    break

            if not has_valid_candidates:
                metrics["_meta"]["no_valid_candidates"] += 1
                continue

            # Process matching results
            metrics["total"]["total"] += 1
            query_results = results_dict.get(query_path.stem, [])

            match_found = any((res[2] >= threshold) and res[4] for res in query_results)

            if match_found:
                metrics["total"]["correct"] += 1
                metrics["_meta"]["successful_matches"] += 1
            else:
                metrics["_meta"]["failed_matches"] += 1

            # Update category metrics
            def update_category(condition, category):
                if condition(query_path.stem):
                    metrics[category]["total"] += 1
                    metrics[category]["correct"] += int(match_found)

            update_category(util_matching.fclt_le_200, "fclt_le_200")
            update_category(util_matching.fclt_200_400, "fclt_200_400")
            update_category(util_matching.fclt_400_800, "fclt_400_800")
            update_category(util_matching.fclt_g_800, "fclt_g_800")
            update_category(util_matching.tilt_l_40, "tilt_l_40")
            update_category(util_matching.tilt_ge_40, "tilt_ge_40")
            update_category(util_matching.cldp_l_40, "cldp_l_40")
            update_category(util_matching.cldp_ge_40, "cldp_ge_40")

        except Exception as e:
            logging.error(f"Error processing {query_path.stem}: {str(e)}")
            continue

    return metrics

def display_metrics(metrics, console):
    """Display metrics with complete breakdown."""
    table = Table(title="[bold]Evaluation Results[/bold]", show_header=True,
                header_style="bold magenta", row_styles=["none", "dim"])
    table.add_column("Category", style="cyan")
    table.add_column("Total", style="green")
    table.add_column("Localized", style="yellow")
    table.add_column("Failed", style="red")
    table.add_column("Success Rate", style="bold blue")

    # Main metrics
    for key in metrics:
        if key.startswith("_"): continue
        
        data = metrics[key]
        total = data["total"]
        success = data["correct"]
        failed = total - success if total > 0 else 0
        rate = (success/total)*100 if total > 0 else 0.0
        
        table.add_row(
            key.replace("_", " ").title(),
            str(total),
            str(success),
            str(failed),
            f"{rate:.1f}%"
        )

    # Add meta statistics
    meta = metrics["_meta"]
    table.add_row(
        "[bold]Queries Without Valid Candidates[/bold]",
        str(meta["no_valid_candidates"]),
        "—",
        "—",
        "—"
    )
    table.add_row(
        "[bold]Total Queries Processed[/bold]",
        str(meta["total_queries"]),
        str(meta["successful_matches"]),
        str(meta["failed_matches"] + meta["no_valid_candidates"]),
        f"{(meta['successful_matches']/meta['total_queries'])*100:.1f}%" if meta["total_queries"] > 0 else "0.0%"
    )

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Evaluate image matching results")
    parser.add_argument("results_file", help="Path to results.torch file")
    parser.add_argument("--data_dir", type=str, default="./data", 
                      help="Original data directory with query folders")
    args = parser.parse_args()

    console = Console()
    try:
        data = load_torch_file(args.results_file)
        threshold = compute_threshold(data)
        metrics = compute_evaluation_metrics(data, threshold, args.data_dir)
        
        console.print(f"\n[bold]Detection Threshold:[/bold] {threshold}")
        console.print(f"[bold]Total Queries:[/bold] {metrics['_meta']['total_queries']}")
        display_metrics(metrics, console)
        
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {str(e)}")

if __name__ == "__main__":
    main()