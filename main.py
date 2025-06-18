import cv2
import sys
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torchvision.transforms as tfm
import time  # <-- Added for timing
import os
sys.path.append(str(Path("image-matching-models")))







# CORRECT this is the only path we need not append the rest gets appended with ir
sys.path.append(str(Path("image-matching-models/matching/third_party")))


print("\n--- sys.path before get_matcher ---")
import pprint
pprint.pprint(sys.path)
print("-----------------------------------\n")


import commons
import util_matching
from matching import get_matcher

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--matcher", type=str, default="sift-lg", help="_")
parser.add_argument("-nk", "--max_num_keypoints", type=int, default=2048, help="_")
parser.add_argument("-ni", "--num_iterations", type=int, default=4, help="_")
parser.add_argument("-is", "--img_size", type=int, default=1024, help="_")
parser.add_argument("--save_images", action="store_true", help="_")
parser.add_argument(
    "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="_"
)
parser.add_argument("--data_dir", type=str, default="./data", help="_")
parser.add_argument(
    "--log_dir",
    type=str,
    default="default",
    help="name of directory on which to save the logs, under logs/log_dir",
)
parser.add_argument(
    "--second_matcher",
    type=str,
    default=None,
    help="You can specify a second matcher which will be used if the first one does not produce enough matches.",
)

args = parser.parse_args()
start_time = datetime.now()

if args.second_matcher:
    log_dir = (
        Path("logs")
        / f"out_{args.matcher}_{args.second_matcher}"
        / f"number_of_iterations_{args.num_iterations}"
        / f"image_size_{args.img_size}"
        / f"number_of_keypoints_{args.max_num_keypoints}"
        / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    )
else:
    log_dir = (
        Path("logs")
        / f"out_{args.matcher}"
        / f"number_of_iterations_{args.num_iterations}"
        / f"image_size_{args.img_size}"
        / f"number_of_keypoints_{args.max_num_keypoints}"
        / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    )

commons.setup_logging(log_dir, stdout="info")
commons.make_deterministic(0)
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {log_dir}")

args.data_dir = Path(args.data_dir)
assert args.data_dir.exists(), f"{args.data_dir} does not exist"

# Initialize matchers
matcher = get_matcher(
    args.matcher, device=args.device, max_num_keypoints=args.max_num_keypoints
)

second_matcher = None
if args.second_matcher:
    second_matcher = get_matcher(
        args.second_matcher,
        device=args.device,
        max_num_keypoints=args.max_num_keypoints,
    )

logging.info(f"Main matcher: {matcher}")
logging.info(f"Secondary matcher: {second_matcher}")

queries_input_folders = sorted(list(args.data_dir.glob("*")))
all_results = []

query_times = []  # <-- Store per-query times

for folder in tqdm(queries_input_folders):
    paths = sorted(list(folder.glob("*")))
    assert len(paths) == 11, "Expected 11 files per folder"
    query_path = paths[0]
    preds_paths = paths[1:]

    query_centerpoint = util_matching.get_centerpoint_from_query_path(query_path)

    query_start_time = time.time()  # <-- Start timing for this query

    for pred_idx, pred_path in enumerate(preds_paths):
        try:
            query_log_dir = log_dir / query_path.stem / f"{pred_idx:02d}"

            # this is a very nasty error which I found to happen 
            # basically the rotation angles which are encoded in the prediction 
            # require the query to be turned clockwise however 
            # by defult tfm.functional.rotate rotates counterclockwise so we need to mulitply by -1
            # this results in +3.6% performance in matchers which are not invariant to rotations e.g master
            rot_angle = int(pred_path.name.split("__")[2].replace("rot", "")) * -1
            assert rot_angle % 90 == 0, "Rotation should be multiple of 90"

            # Load and prepare images
            query_image = matcher.image_loader(query_path, args.img_size).to(
                args.device
            )
            query_image = tfm.functional.rotate(query_image, rot_angle)
            surrounding_image = matcher.image_loader(pred_path, args.img_size * 3).to(
                args.device
            )
            pred_footprint = util_matching.path_to_footprint(pred_path)

            # Save input images if requested
            if args.save_images:
                query_log_dir.mkdir(exist_ok=True, parents=True)
                tfm.ToPILImage()(query_image).save(query_log_dir / query_path.name)
                tfm.ToPILImage()(surrounding_image).save(
                    query_log_dir / "surrounding_img.jpg"
                )

            fm = None
            final_result = None
            used_secondary = False

            # Try with primary matcher first
            for matcher_idx, current_matcher in enumerate([matcher, second_matcher]):
                if current_matcher is None:
                    continue

                matcher_suffix = f"_m{matcher_idx+1}"
                found_match = True
                local_fm = None  # Reset FM for each matcher

                for iteration in range(args.num_iterations):
                    viz_params = (
                        {
                            "output_dir": query_log_dir,
                            "output_file_suffix": f"{iteration}{matcher_suffix}",
                            "query_path": query_path,
                            "pred_path": query_log_dir
                            / f"pred_{iteration}{matcher_suffix}.jpg",
                        }
                        if args.save_images
                        else None
                    )

                    num_inliers, local_fm, predicted_footprint, pretty_footprint = (
                        util_matching.estimate_footprint(
                            local_fm,
                            query_image,
                            surrounding_image,
                            current_matcher,
                            pred_footprint,
                            HW=args.img_size,
                            save_images=args.save_images,
                            viz_params=viz_params,
                        )
                    )

                    if num_inliers == -1:
                        found_match = False
                        logging.debug(
                            f"{query_path.stem} {pred_idx=} matcher{matcher_idx+1} {iteration=:02d} NOT_FOUND"
                        )
                        break

                    # Polygon processing
                    pred_polygon = util_matching.get_polygon(
                        predicted_footprint.numpy()
                    )
                    pred_polygon = util_matching.enlarge_polygon(pred_polygon, 3)

                    # Result evaluation
                    is_correct = pred_polygon.contains(query_centerpoint)
                    log_msg = {
                        "stem": query_path.stem,
                        "pred_idx": pred_idx,
                        "matcher": f"matcher{matcher_idx+1}",
                        "iteration": iteration,
                        "inliers": num_inliers,
                        "correct": is_correct,
                        "footprint": pretty_footprint,
                    }
                    print(log_msg)

                    # Store final result if last iteration
                    if iteration == args.num_iterations - 1:
                        final_result = (
                            query_path.stem,
                            pred_idx,
                            num_inliers,
                            predicted_footprint,
                            is_correct,
                        )

                if found_match:
                    fm = local_fm  # Preserve FM for potential reuse
                    used_secondary = matcher_idx == 1
                    break

            # Record final result
            if final_result:
                all_results.append(final_result + (used_secondary,))
            else:
                print(f"{query_path.stem} {pred_idx=} ALL_MATCHERS_FAILED")

        except (
            ValueError,
            torch._C._LinAlgError,
            cv2.error,
            IndexError,
            AttributeError,
        ) as e:
            print(f"Error processing {query_path.stem} {pred_idx=}: {str(e)}")

    query_end_time = time.time()  
    query_duration = query_end_time - query_start_time
    query_times.append(query_duration) 

torch.save(all_results, log_dir / "results.torch")

# ... (rest of your code, e.g. thresholding, stats, etc.)

# At the end, display the average seconds per query
if query_times:
    avg_seconds_per_query = sum(query_times) / len(query_times)
    print(f"Average seconds per query: {avg_seconds_per_query:.3f}")
    logging.info(f"Average seconds per query: {avg_seconds_per_query:.3f}")
else:
    print("No queries were processed.")
    logging.info("No queries were processed.")
