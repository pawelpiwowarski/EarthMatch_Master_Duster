
# Modified EarthMatch 


## Summary of Code Modifications

This document outlines the key modifications made to the original codebase. These changes introduce new features, improve performance, and ensure compatibility with newer versions of PyTorch.


### 0. How to use new models?

You need to make sure you have downloaded all of the models and their dependancies in the `./image-matching-models/matching/third-party` f.e Duster,Master,croco etc. 
The script will automatically take care of the imports as long as they are there. 


### 1. New Feature: Ensemble of Models

To improve the robustness of the matching pipeline, the main script has been updated to support a secondary model.

-   **File Modified:** `main.py`
-   **Change:** The script can now accept a second model as an argument.
-   **Functionality:** If the primary model fails to compute a homography for a given image pair, the system will automatically fall back to using the second model to attempt the computation. This prevents a complete failure when the primary model encounters a difficult case.

### 2. Bug Fixes and Performance Improvements

#### Corrected Query Image Rotation

We saw that sometimes  the orientation of query images did not align with the candidate images, which negatively impacted matching performance.

-   **Change:** The rotation angle applied to query images is now multiplied by -1.
-   **Effect:** This changes the rotation from counter-clockwise to clockwise.
-   **Reason:** This ensures that the query and candidate images share the same orientation, leading to more accurate keypoint matching and better overall results.

#### Enhanced Keypoint Matching for MASTER

To allow the MASTER model to use the maximal number of keypoints paramater. 

-   **Files to Replace:** `/path/to/image-matching-models/third-party/mast3r/fast_nn.py` `image-matching-models/matching/__init__.py`
-   **Action:** Replace the existing `fast_nn.py` file with the modified version provided in `modifications/fast_nn.py`. In the `__init__.py` the `image-matching-models/matching/__init__.py` pass the max_num_keypoints to     `elif matcher_name in ["master", "mast3r"]:
        from matching.im_models import master

        return master.Mast3rMatcher(device, max_num_keypoints, *args, **kwargs)

    elif matcher_name == "doghardnet-nn"` :
-   **Reason:** This change facilitates matching with a maximal number of keypoints, such that it can be directly compared to other models.

### 3. PyTorch Compatibility Updates

These changes address warnings and errors that appear with recent versions of PyTorch, ensuring the code runs smoothly on modern environments.

#### Duster `autocast` Deprecation Warning

Newer PyTorch versions recommend a more explicit API for automatic mixed precision.

-   **File Modified:** `duster/inference.py`
-   **Change:**
    ```diff
    - torch.cuda.amp.autocast(enabled=bool(use_amp))
    + torch.amp.autocast(device_type=device, enabled=bool(use_amp))
    ```
-   **Reason:** This update uses the current recommended `torch.amp.autocast` syntax, which suppresses deprecation warnings and ensures future compatibility.

#### `torch.load` Security Update Error

A recent security update in PyTorch changed the default behavior of `torch.load`. By default, it now assumes it is only loading model weights (`weights_only=True`), which can cause errors if the checkpoint file also contains other data (like optimizer states or metadata).

-   **Files Modified:** `duster/model.py` and `master/model.py`
-   **Change:**
    ```diff
    - ckpt = torch.load(model_path, map_location='cpu')
    + ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    ```
-   **Reason:** Explicitly setting `weights_only=False` informs PyTorch that the checkpoint is trusted and may contain more than just model weights, resolving the loading error in newer versions.

## Results 

See results of Roma, Master, Duster, Xfeet*,Xfeet_steerers in [Results](./documentation/results.md). To reproduce simply pass the required arguments to the main.py f.e to train Roma + Xfeet_steerers ` python main.py --matcher xfeat-star-steerers-learned --second_matcher roma -nk 1024 -is 512 --device cuda`
# EarthMatch

EarthMatch (CVPR 2024 IMW) is an image-matching / coregistration pipeline used to localize photos taken by astronauts aboard the ISS. It takes as input a pair of images, the astronaut photo to be localized and a potential candidate (obtain from a retrieval method like EarthLoc) and, if the two images do overlap, it outputs their precise coregistration.

<p float="left">
  <img src="https://earthloc-and-earthmatch.github.io/static/images/pipeline.png" />
</p>

[Check out our webpage](https://earthloc-and-earthmatch.github.io/)

The paper, called "EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography" is accepted to the 2024 CVPR workshop of "Image Matching: Local Features & Beyond 2024". Below you can see how the iterative coregistration takes place (4 iterations, num keypoints usually increasing with more iterations).

<p float="left">
  <img src="https://earthloc-and-earthmatch.github.io/static/images/qualitative/gifs/gif1.gif" height="150" />
  <img src="https://earthloc-and-earthmatch.github.io/static/images/qualitative/gifs/gif2.gif" height="150" />
  <img src="https://earthloc-and-earthmatch.github.io/static/images/qualitative/gifs/gif3.gif" height="150" />
  <img src="https://earthloc-and-earthmatch.github.io/static/images/qualitative/gifs/gif4.gif" height="150" />
</p>

## Run the experiments

```
# Clone the repo
git clone --recursive https://github.com/gmberton/EarthMatch
cd EarthMatch
# Download the data
rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/EarthMatch/data .
# Run the experiment with SIFT-LightGlue
python main.py --matcher sift-lg --max_num_keypoints 2048 --img_size 512 --data_dir data --log_dir out_sift-lg --save_images
```

The data contains 268 astronaut photos and, for each of them, the top-10 predictions obtained from a worldwide database with an enhanced version of EarthMatch.

The logs and visualizations will be automatically saved in `./logs/out_sift-lg` (note that using `--save_images` will save images for all results and slow down the experiment.

You can set the matcher to any of the 17 matchers used in the [image-matching-models repo](https://github.com/gmberton/image-matching-models).


## Cite
```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Berton, Gabriele and Goletto, Gabriele and Trivigno, Gabriele and Stoken, Alex and Caputo, Barbara and Masone, Carlo},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
