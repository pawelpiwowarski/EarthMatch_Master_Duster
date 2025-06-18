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
import matplotlib.pyplot as plt
from PIL import Image
import math

sys.path.append(str(Path('image-matching-models')))
sys.path.append(str(Path('image-matching-models/third_party/RoMa')))
sys.path.append(str(Path('image-matching-models/third_party/duster')))
sys.path.append(str(Path('image-matching-models/third_party/DeDoDe')))
sys.path.append(str(Path('image-matching-models/third_party/Steerers')))
sys.path.append(str(Path('image-matching-models/third_party/Se2_LoFTR')))
sys.path.append(str(Path('image-matching-models/third_party/LightGlue')))
sys.path.append(str(Path('image-matching-models/third_party/imatch-toolbox')))

import util_matching

from pyproj import Geod
geod = Geod(ellps="WGS84")

def plot_unlocalized_queries(unlocalized_paths, data_dir, results_dict, save_path="plots/unlocalized_queries.png"):
    """
    Create a two-column layout showing only the unlocalized query images.
    If odd number of images, the last image is centered.
    Display cloud percentage, tilt, and focal length in the title.
    """
    # Create plots directory if it doesn't exist
    Path("plots").mkdir(exist_ok=True)
    
    n = len(unlocalized_paths)
    if n == 0:
        return
        
    # Calculate grid dimensions
    cols = 2
    rows = math.ceil(n / cols)
    
    # Create figure
    fig = plt.figure(figsize=(12, 4*rows))
    plt.suptitle("Failure Cases  - Queries Not Localised by None of the Models", fontsize=16, y=1.02)
    
    # Create a gridspec that allows for centered last image
    from matplotlib import gridspec
    if n % 2 == 1:  # Odd number of images
        gs = gridspec.GridSpec(rows, 2)
    
    # Plot each query image
    for idx, query_path in enumerate(unlocalized_paths):
        try:
            # Extract metadata from path
            parts = str(query_path).split("@")
            tilt = parts[5]
            fclt = parts[6]
            cldp = parts[7]
            
            # Handle the last image differently if odd number of images
            if idx == n - 1 and n % 2 == 1:
                # Center the last image by spanning both columns
                ax = plt.subplot(gs[-1, :])
            else:
                # Normal subplot for other images
                ax = plt.subplot(rows, cols, idx + 1)
            
            query_img = Image.open(query_path)
            plt.imshow(query_img)
            plt.axis('off')
            plt.title(f"Tilt: {tilt}°, FL: {fclt}mm, Cloud: {cldp}%", 
                     fontsize=8, wrap=True)
            
        except Exception as e:
            print(f"Error plotting image {query_path}: {str(e)}")
            continue
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[green]Plot of unlocalized queries saved to: {save_path}[/green]")






def load_torch_file(file_path):
    """Load a .torch file using PyTorch."""
    try:
        return torch.load(file_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {str(e)}")

def compute_threshold(all_results, lr_thresh):
    """Calculate the optimal inlier threshold."""
    tp_inliers = [res[2] for res in all_results if res[4]]
    fp_inliers = [res[2] for res in all_results if not res[4]]
    return -1 if not fp_inliers else util_matching.compute_threshold(tp_inliers, fp_inliers, 0.999)

def compute_area_of_a_footprint(polygon):
    coords = np.array(polygon.exterior.coords)
    lat, lon = coords[:, 0], coords[:, 1]
    area_m2, _ = geod.polygon_area_perimeter(lon, lat)
    area_km2 = abs(area_m2) / 1e6 
    return area_km2

def compute_threshold_inliers_and_area(all_results, lr_thresh):
    tp_inliers = [res[2] for res in all_results if res[4]]
    fp_inliers = [res[2] for res in all_results if not res[4]]
    tp_area = [compute_area_of_a_footprint(util_matching.get_polygon(res[3].numpy())) for res in all_results if res[4]]
    fp_area = [compute_area_of_a_footprint(util_matching.get_polygon(res[3].numpy())) for res in all_results if not res[4]]

    inlier_thresh = -1 if not fp_inliers else util_matching.compute_threshold(tp_inliers, fp_inliers, lr_thresh)
    area_thresh = -1 if not fp_area else util_matching.compute_threshold(tp_area, fp_area, lr_thresh)
    return inlier_thresh, area_thresh

def compute_evaluation_metrics(
    all_results, threshold, data_dir, use_area_threshold=False, area_threshold=None, save_images = False
):
    """
    Compute metrics while properly accounting for all queries and generate visualization 
    of unlocalized queries.
    """
    # Create results dictionary for easy lookup
    results_dict = defaultdict(list)
    for res in all_results:
        results_dict[res[0]].append(res)

    query_folders = sorted(list(Path(data_dir).glob("*")))
    total_queries = len(query_folders)
    
    categories = [
        "total",
        "fclt_le_200",
        "fclt_200_400",
        "fclt_400_800",
        "fclt_g_800",
        "tilt_l_40",
        "tilt_ge_40",
        "cldp_l_40",
        "cldp_ge_40",
    ]
    
    # Initialize metrics dictionary
    metrics = {
        category: {
            "correct": 0, 
            "total_localizable": 0, 
            "failed_unlocalizable": 0
        } for category in categories
    }
    metrics["_meta"] = {
        "total_queries": total_queries,
        "successful_matches": 0,
        "failed_matches": 0,
        "false_positives": 0,
        "false_positives_before_threshold": 0,
        "true_positives": 0,
        "true_positives_before_threshold": 0,
    }

    # Track unlocalized queries with viable candidates
    unlocalized_with_candidates = []
    query_details = {}  # Store details for visualization

    for query_folder in tqdm(query_folders, desc="Processing queries"):
        paths = sorted(list(query_folder.glob("*")))
        if not paths:
            continue
            
        query_path = paths[0]
        query_id = query_path.stem

        try:
            gt_center = util_matching.get_centerpoint_from_query_path(query_path)
            has_valid_candidates = False
            valid_candidate_paths = []
            
            # Check all candidate images in folder
            for candidate_path in query_folder.glob("*"):
                if candidate_path == query_path:
                    continue
                
                candidate_footprint = util_matching.path_to_footprint(candidate_path)
                pred_polygon = util_matching.get_polygon(candidate_footprint.numpy())

                if pred_polygon.contains(gt_center):
                    has_valid_candidates = True
                    valid_candidate_paths.append(candidate_path)

            # Determine categories for this query
            query_categories = ["total"]
            if util_matching.fclt_le_200(query_id):
                query_categories.append("fclt_le_200")
            if util_matching.fclt_200_400(query_id):
                query_categories.append("fclt_200_400")
            if util_matching.fclt_400_800(query_id):
                query_categories.append("fclt_400_800")
            if util_matching.fclt_g_800(query_id):
                query_categories.append("fclt_g_800")
            if util_matching.tilt_l_40(query_id):
                query_categories.append("tilt_l_40")
            if util_matching.tilt_ge_40(query_id):
                query_categories.append("tilt_ge_40")
            if util_matching.cldp_l_40(query_id):
                query_categories.append("cldp_l_40")
            if util_matching.cldp_ge_40(query_id):
                query_categories.append("cldp_ge_40")

            # Update categories based on valid candidates status
            for category in query_categories:
                if has_valid_candidates:
                    metrics[category]["total_localizable"] += 1
                else:
                    metrics[category]["failed_unlocalizable"] += 1

            if not has_valid_candidates:
                continue

            # Process matching results
            query_results = results_dict.get(query_id, [])
            match_found = False
            best_inliers = -1
            best_candidate = None
            
            for res in query_results:
                inlier_ok = (res[2] >= threshold)
                if res[2] > best_inliers:
                    best_inliers = res[2]
                    best_candidate = res[1]  # Assuming res[1] contains candidate path
                
                if use_area_threshold and area_threshold is not None:
                    area = compute_area_of_a_footprint(
                        util_matching.get_polygon(res[3].numpy())
                    )
                    area_ok = (area >= area_threshold)
                    if (inlier_ok or area_ok) and res[4]:
                        match_found = True
                        break
                else:
                    if inlier_ok and res[4]:
                        match_found = True
                        break

            # Store query details for unlocalized cases
            if (not match_found) and has_valid_candidates:
                unlocalized_with_candidates.append(str(query_path))
                query_details[query_id] = {
                    'query_path': query_path,
                    'best_candidate': best_candidate,
                    'best_inliers': best_inliers,
                    'valid_candidates': valid_candidate_paths
                }

            # Update metrics based on match status
            if match_found:
                metrics["_meta"]["successful_matches"] += 1
                for category in query_categories:
                    metrics[category]["correct"] += 1
            else:
                metrics["_meta"]["failed_matches"] += 1

        except Exception as e:
            logging.error(f"Error processing {query_id}: {str(e)}")
            continue

    # Count false/true positives before and after threshold
    false_positives = 0
    false_positives_before_threshold = 0
    true_positives = 0
    true_positives_before_threshold = 0
    
    for res in all_results:
        if res[4]:
            true_positives_before_threshold += 1
        else:
            false_positives_before_threshold += 1
            
        inlier_ok = (res[2] >= threshold)
        if use_area_threshold and area_threshold is not None:
            area = compute_area_of_a_footprint(
                util_matching.get_polygon(res[3].numpy())
            )
            area_ok = (area >= area_threshold)
            if (inlier_ok or area_ok):
                if res[4]:
                    true_positives += 1
                else:
                    false_positives += 1
        else:
            if inlier_ok:
                if res[4]:
                    true_positives += 1
                else:
                    false_positives += 1

    # Update meta statistics
    metrics["_meta"].update({
        "false_positives": false_positives,
        "false_positives_before_threshold": false_positives_before_threshold,
        "true_positives": true_positives,
        "true_positives_before_threshold": true_positives_before_threshold,
        "no_valid_candidates": metrics["total"]["failed_unlocalizable"],
        "total_positives": metrics["total"]["total_localizable"],
        "false_negatives": metrics["_meta"]["failed_matches"]
    })

    # Handle unlocalized queries visualization
    if unlocalized_with_candidates and save_images:
        print("\n[red]Queries with viable candidates but NOT localized:[/red]")
        for path in unlocalized_with_candidates:
            print(path)
        # Create and save plot with side-by-side comparison
        plot_unlocalized_queries(unlocalized_with_candidates, data_dir, results_dict)
    else:
        print("\n[green]All queries with viable candidates were localized.[/green]")

    return metrics


def display_metrics(metrics, console, inlier_threshold=None, area_threshold=None):
    """Display metrics with new columns for failed categories."""
    title = "[bold]Evaluation Results[/bold]"
    if area_threshold is not None:
        title += f" (Area Threshold: {area_threshold:.2f})"
    elif inlier_threshold is not None:
        title += f" (Inlier Threshold: {inlier_threshold:.2f})"

    table = Table(title=title, show_header=True,
                 header_style="bold magenta", row_styles=["none", "dim"])
    table.add_column("Category", style="cyan")
    table.add_column("Total Localizable", style="green")
    table.add_column("Localized", style="yellow")
    table.add_column("Failed (L)", style="red")
    table.add_column("Failed (U)", style="red")
    table.add_column("Success Rate", style="bold blue")

    # Main metrics
    categories = [
        "total",
        "fclt_le_200",
        "fclt_200_400",
        "fclt_400_800",
        "fclt_g_800",
        "tilt_l_40",
        "tilt_ge_40",
        "cldp_l_40",
        "cldp_ge_40",
    ]
    for key in categories:
        data = metrics[key]
        total_localizable = data["total_localizable"]
        localized = data["correct"]
        failed_L = total_localizable - localized
        failed_U = data["failed_unlocalizable"]
        rate = (localized / total_localizable * 100) if total_localizable > 0 else 0.0
        
        table.add_row(
            key.replace("_", " ").title(),
            str(total_localizable),
            str(localized),
            str(failed_L),
            str(failed_U),
            f"{rate:.1f}%"
        )

    # Add meta statistics
    meta = metrics["_meta"]
    total_cat = metrics["total"]
    total_queries = meta["total_queries"]
    failed_L_total = total_cat["total_localizable"] - total_cat["correct"]
    failed_U_total = total_cat["failed_unlocalizable"]
    success_rate_total = (total_cat["correct"] / total_cat["total_localizable"] * 100) if total_cat["total_localizable"] > 0 else 0.0

    table.add_row(
        "[bold]Total Queries Processed[/bold]",
        str(total_queries), "—", "—", "—",
        f"{(total_cat['correct'] / total_queries * 100) if total_queries >0 else 0:.1f}%"
    )

    console.print(table)
        # Print false positives before and after thresholding
        # Print true/false positives before and after thresholding
    console.print(
        f"[bold green]True Positives (before thresholding):[/bold green] "
        f"{metrics['_meta']['true_positives_before_threshold']}"
    )
    console.print(
        f"[bold green]True Positives (after thresholding):[/bold green] "
        f"{metrics['_meta']['true_positives']}"
    )
    console.print(
        f"[bold red]False Positives (before thresholding):[/bold red] "
        f"{metrics['_meta']['false_positives_before_threshold']}"
    )
    console.print(
        f"[bold red]False Positives (after thresholding):[/bold red] "
        f"{metrics['_meta']['false_positives']}"
    )





def main():
    parser = argparse.ArgumentParser(description="Evaluate image matching results")
    parser.add_argument("results_file", help="Path to results.torch file")
    parser.add_argument("--data_dir", type=str, default="./data", 
                      help="Original data directory with query folders")
    parser.add_argument("--use_area_threshold", action="store_true",
                      help="If set, use area threshold in addition to inlier threshold")
    parser.add_argument("--lr_thresh", type=float,default=0.999, help="The confidence threshold used when computing the LR model, to compute the downstream thresholds" )
    parser.add_argument("--save_images", action="store_true", help='Whether to save the images which are not localizable')
    args = parser.parse_args()

    console = Console()
    try:
        data = load_torch_file(args.results_file)
        if args.use_area_threshold:
            inlier_thresh, area_thresh = compute_threshold_inliers_and_area(data, args.lr_thresh)
            threshold = inlier_thresh
        else:
            threshold = compute_threshold(data, args.lr_thresh)
            area_thresh = None
        metrics = compute_evaluation_metrics(
            data, threshold, args.data_dir, 
            use_area_threshold=args.use_area_threshold, 
            area_threshold=area_thresh,
            save_images=args.save_images
        )
        
        console.print(f"\n[bold]Detection Threshold:[/bold] {threshold}")
        if args.use_area_threshold:
            console.print(f"[bold]Area Threshold:[/bold] {area_thresh}")
        console.print(f"[bold]Total Queries:[/bold] {metrics['_meta']['total_queries']}")
        display_metrics(metrics, console)
        
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {str(e)}")



if __name__ == "__main__":
    main()