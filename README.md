
# LiDAR-3D-Detection-Robustness-SFA3D

**Evaluating the robustness of LiDAR-based 3D object detection under point-cloud degradation using SFA3D and the KITTI dataset.**

This project evaluates how LiDAR point-cloud degradation affects a pretrained **SFA3D 3D object detector**. Using 500 KITTI frames, I compared detection performance under clean input, random point dropout, and Gaussian coordinate noise.

Rather than training a new model, the project focuses on **reproducible robustness evaluation, quantitative analysis, and frame-level failure investigation** for autonomous-driving perception.

## Key Results

- **500 KITTI frames** evaluated under **7 input conditions**.
- **3,500 frame-condition evaluations** covering clean input, three point-dropout levels, and three coordinate-noise levels.
- Evaluated detection precision, recall, and matched 3D intersection over union (IoU).
- Investigated individual frames where detections that matched ground-truth vehicles under clean input failed under coordinate noise.

| Condition               | Pooled Precision | Pooled Recall | Mean Matched 3D IoU |
|:------------------------|-----------------:|--------------:|--------------------:|
| Clean                   | 0.5404           | 0.5869        | 0.6649              |
| Random dropout 20%      | 0.5224           | 0.5680        | 0.6597              |
| Random dropout 40%      | 0.5020           | 0.5436        | 0.6567              |
| Random dropout 60%      | 0.5093           | 0.4991        | 0.6482              |
| Coordinate noise 0.05 m | 0.5028           | 0.5461        | 0.6552              |
| Coordinate noise 0.10 m | 0.4955           | 0.5070        | 0.6360              |
| Coordinate noise 0.20 m | 0.4829           | 0.2239        | 0.6057              |

**Key observation:** Pooled recall decreased from 0.5869 under clean input to 0.2239 with 0.20 m Gaussian coordinate noise—a decrease of approximately **36.3 percentage points**. This indicates a substantial degradation in detection performance under the tested perturbation.

*These are results from a custom evaluation on a selected 500-frame KITTI subset, not official KITTI benchmark scores.*

<img width="3270" height="1616" alt="robustness_metrics" src="https://github.com/user-attachments/assets/81d60cc1-a9f6-4e98-802d-4c3ae9a2070d" />

## Project Overview

SFA3D detects 3D objects from LiDAR point clouds by converting the points into a Bird's-Eye View (BEV) representation and predicting object centers, dimensions, orientations, and positions.

Building on the pretrained detector, I developed an evaluation workflow to compare its predictions under controlled input degradation and examine where detection performance deteriorates.

### Evaluation Conditions

| Condition                 | Configuration |
|:--------------------------|:--------------|
| Clean                     | Original LiDAR point cloud |
| Random point dropout      | 20%, 40%, and 60% of points removed |
| Gaussian coordinate noise | Independent XYZ perturbations with standard deviations of 0.05 m, 0.10 m, and 0.20 m |

Random dropout simulates reduced point availability, while coordinate noise tests sensitivity to spatial perturbations. These are **synthetic corruption experiments**, not measurements collected from adverse real-world weather or faulty sensors.

## Methodology

The evaluation pipeline consists of the following stages:

1. **Load KITTI data:** Read LiDAR point clouds and vehicle annotations.
2. **Apply controlled degradation:** Generate clean, point-dropout, or coordinate-noise inputs using reproducible random seeds.
3. **Preprocess LiDAR:** Apply SFA3D's configured spatial boundaries and convert the filtered point cloud into a BEV representation.
4. **Run pretrained inference:** Pass the BEV input through the FPN-ResNet-18-based SFA3D model.
5. **Decode detections:** Apply heatmap-based decoding, post-processing, and conversion to metric LiDAR coordinates.
6. **Evaluate predictions:** Match predicted car boxes to ground-truth car boxes using 3D IoU and one-to-one assignment.
7. **Analyze failures:** Compare results for the same frames under clean and corrupted conditions.

## Evaluation Protocol

The main experiment evaluates 500 frames under seven conditions, resulting in 3,500 frame-condition records.

- **Target class:** Car.
- **Matching threshold:** 3D IoU ≥ 0.50.
- **Assignment:** One-to-one matching between predictions and ground-truth boxes using the Hungarian assignment algorithm.
- **True positives (TP):** Matched predicted and ground-truth boxes.
- **False positives (FP):** Unmatched predicted boxes.
- **False negatives (FN):** Unmatched ground-truth boxes.

The metrics are calculated as:

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$

Pooled precision and recall are calculated from the total TP, FP, and FN counts across the evaluated frames. Mean matched 3D IoU summarizes the overlap of matched boxes.

## Frame-Level Failure Analysis

In addition to aggregate metrics, I examined individual frames where the detector performed well on clean input but struggled after coordinate perturbation.

For example, in KITTI frame `000308`, two car detections matched their respective ground-truth boxes under clean input. With 0.20 m coordinate noise, the remaining car prediction had an IoU below the 0.50 matching threshold for both ground-truth vehicles, resulting in zero true positives for that frame.

This paired analysis helps connect aggregate metric changes to specific detection failures.

<img width="4770" height="2190" alt="frame_000308_failure" src="https://github.com/user-attachments/assets/b65ace20-f670-4081-825a-1ed95b296eca" />


## Model and Implementation

This project uses the pretrained SFA3D implementation with an FPN-ResNet-18 backbone. It does not introduce a new model architecture or retrain the detector as part of the main robustness experiment.

### Main Components

- **Framework:** Python and PyTorch
- **Dataset:** KITTI 3D object detection dataset
- **Model:** SFA3D with FPN-ResNet-18
- **Point-cloud processing:** NumPy
- **Visualization:** Matplotlib, OpenCV, and Plotly
- **Matching:** SciPy's linear assignment algorithm
- **3D box overlap:** Custom oriented 3D IoU calculation

The SFA3D inference workflow includes:

- LiDAR spatial filtering and BEV generation
- Height, intensity, and density feature maps
- Center heatmap and box-parameter prediction
- Top-K detection decoding and confidence filtering
- Conversion of detections into metric LiDAR coordinates
- 3D bounding-box visualization and evaluation

```

## Limitations

- Results are based on a selected 500-frame subset, not the full KITTI benchmark.
- The corruption conditions are synthetic and do not reproduce every characteristic of real sensor degradation or adverse weather.
- The reported metrics use a custom evaluation protocol and should not be compared directly with official KITTI leaderboard scores.
- Results depend on the pretrained checkpoint, preprocessing, coordinate conventions, IoU implementation, and matching threshold.

## Acknowledgments

The detector and its core inference components are based on the open-source [SFA3D implementation by maudzung](https://github.com/maudzung/SFA3D).

This repository documents my inference setup, robustness evaluation, quantitative analysis, and visualization work built around that implementation.
