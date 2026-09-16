# 3D Object Detection with SFA3D


This project focuses on understanding and running the SFA3D inference pipeline, including LiDAR preprocessing, Bird's-Eye View (BEV) generation, neural network inference, heatmap-based detection decoding, 3D bounding-box conversion, and visualization of predicted objects.

The implementation was built as a **hands-on computer vision and autonomous driving practice project** to better understand LiDAR-based 3D object detection and BEV representations.

<img width="605" height="607" alt="image" src="https://github.com/user-attachments/assets/58fc47de-459f-45f4-b6e0-73da24f35432" />


---

## Overview

SFA3D performs 3D object detection directly from LiDAR point clouds by transforming the point cloud into a **Bird's-Eye View representation** and predicting 3D object properties from that representation.

The workflow implemented in this project is:

```text
KITTI LiDAR (.bin)
        │
        ▼
LiDAR Point Cloud
        │
        ▼
Spatial Filtering
        │
        ▼
SFA3D BEV Representation
        │
        ├── Height Map
        ├── Intensity Map
        └── Density Map
        │
        ▼
FPN-ResNet-18
        │
        ▼
Detection Heads
        │
        ├── Center Heatmap
        ├── Center Offset
        ├── Direction
        ├── Z Coordinate
        └── Dimensions
        │
        ▼
Top-K Decoding
        │
        ▼
Post-Processing
        │
        ▼
3D Bounding Boxes
        │
        ▼
KITTI/LiDAR Coordinates
        │
        ▼
BEV Visualization
```

---

## What I Practiced

This project provided hands-on experience with:

* LiDAR point-cloud processing
* KITTI 3D object detection data
* Bird's-Eye View representations
* Point-cloud spatial filtering
* BEV feature-map construction
* Center-based 3D object detection
* FPN-ResNet architectures
* Heatmap-based object localization
* Top-K detection decoding
* 3D bounding-box parameterization
* LiDAR coordinate systems
* 3D object visualization
* PyTorch inference
* OpenCV and Matplotlib visualization
* Understanding the relationship between BEV coordinates and real-world LiDAR coordinates

---

## SFA3D Model

The project uses the SFA3D implementation from:

**GitHub:**
https://github.com/Hasnt-Ahmad/3D-Object-Detection-SFA3D

The model is constructed using an **FPN-ResNet-18** backbone.

```python
model = get_pose_net(
    num_layers=18,
    heads=heads,
    head_conv=64,
    imagenet_pretrained=False
)
```

The detection heads are:

```python
heads = {
    'hm_cen': 3,
    'cen_offset': 2,
    'direction': 2,
    'z_coor': 1,
    'dim': 3
}
```

These heads predict the information required to reconstruct 3D detections.

### Detection Heads

| Head         | Output | Purpose                 |
| ------------ | -----: | ----------------------- |
| `hm_cen`     |      3 | Object-center heatmap   |
| `cen_offset` |      2 | Sub-pixel center offset |
| `direction`  |      2 | Object orientation      |
| `z_coor`     |      1 | Vertical position       |
| `dim`        |      3 | 3D object dimensions    |

The model uses three object classes, consistent with the SFA3D configuration.

---

## Pretrained Model

Inference uses the pretrained checkpoint:

```text
checkpoints/
└── fpn_resnet_18/
    └── fpn_resnet_18_epoch_300.pth
```

The checkpoint is loaded with PyTorch:

```python
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)
```

# LiDAR Preprocessing

Before creating the BEV representation, the point cloud is filtered using the spatial boundaries defined by SFA3D.

The filtering checks:

```text
minX <= x <= maxX
minY <= y <= maxY
minZ <= z <= maxZ
```

The implementation uses:

```python
points = get_filtered_lidar(
    points,
    cnf.boundary
)
```

The vertical coordinate is then shifted:

```python
lidar[:, 2] = lidar[:, 2] - minZ
```

This places the minimum height boundary at zero for BEV generation.

---

# Bird's-Eye View Representation

One of the main concepts explored in this project is converting a 3D LiDAR point cloud into a **2D BEV feature map**.

SFA3D divides the LiDAR space into a regular grid according to the configured discretization.

```python
PointCloud[:, 0] = np.int_(
    np.floor(
        PointCloud[:, 0] /
        cnf.DISCRETIZATION
    )
)

PointCloud[:, 1] = np.int_(
    np.floor(
        PointCloud[:, 1] /
        cnf.DISCRETIZATION
    )
    + Width / 2
)
```

Each LiDAR point is therefore mapped from continuous real-world coordinates into BEV grid coordinates.

---

## BEV Feature Channels

The BEV representation contains three feature maps:

### 1. Height Map

Represents the height of the highest point in each BEV cell.

```python
heightMap[
    point_x,
    point_y
] = point_z / max_height
```

### 2. Intensity Map

Stores the LiDAR reflectance/intensity:

```python
intensityMap[
    point_x,
    point_y
] = point_intensity
```

### 3. Density Map

Represents the number of points occupying each cell.

The density is normalized using:

```python
normalizedCounts = np.minimum(
    1.0,
    np.log(unique_counts + 1) /
    np.log(64)
)
```

These three maps are combined into the SFA3D BEV representation:

```text
BEV
├── Height
├── Intensity
└── Density
```

The resulting representation has the structure:

```text
[3, BEV_HEIGHT, BEV_WIDTH]
```

---

# Detection

After BEV generation, the tensor is passed to the pretrained SFA3D network.

```python
outputs = model(bev)
```

The center heatmap is converted from logits into probabilities:

```python
hm_cen = _sigmoid(
    outputs['hm_cen']
)
```

The model outputs are then decoded using the SFA3D decoding procedure.

```python
detections = decode(
    hm_cen,
    outputs['cen_offset'],
    outputs['direction'],
    outputs['z_coor'],
    outputs['dim'],
    K=40
)
```

The decoder selects the top `K=40` candidate object centers.

---

# Detection Decoding

The decoded detection representation contains:

```text
[scores,
 x,
 y,
 z,
 h,
 w,
 l,
 direction,
 class]
```

The center heatmap provides candidate object locations.

The center offset refines the discretized BEV location:

```python
xs = xs + cen_offset[:, :, 0:1]
ys = ys + cen_offset[:, :, 1:2]
```

The network additionally predicts:

* vertical position
* object dimensions
* object orientation
* object class

These predictions are combined into the final detection tensor.

---

# Post-Processing

After decoding, SFA3D's post-processing step is applied:

```python
detections = post_processing(
    detections,
    num_classes=3,
    down_ratio=4,
    peak_thresh=0.20
)
```

The configured peak threshold is:

```text
0.20
```

This removes low-confidence candidate detections from the decoded results.

---

# Converting BEV Predictions to LiDAR Coordinates

The decoded predictions initially use BEV/grid coordinates.

The project converts these predictions back into real-world LiDAR coordinates using:

```python
convert_det_to_real_values(
    detections,
    num_classes=3
)
```

For example, the BEV coordinate is converted using the configured BEV dimensions and physical boundary size:

```python
x = (
    _y / cnf.BEV_HEIGHT
    * cnf.bound_size_x
    + cnf.boundary['minX']
)

y = (
    _x / cnf.BEV_WIDTH
    * cnf.bound_size_y
    + cnf.boundary['minY']
)
```

The vertical coordinate is reconstructed as:

```python
z = (
    _z +
    cnf.boundary['minZ']
)
```

Object dimensions are similarly converted from BEV units into physical dimensions:

```python
w = (
    _w /
    cnf.BEV_WIDTH
    * cnf.bound_size_y
)

l = (
    _l /
    cnf.BEV_HEIGHT
    * cnf.bound_size_x
)
```

The resulting representation is:

```text
[class_id, x, y, z, h, w, l, yaw]
```

---

# 3D Bounding Box Construction

For visualization, every detection is converted into its eight 3D corners.

The box is represented by:

```text
[class_id, x, y, z, h, w, l, yaw]
```

where:

* `x` — object center along the forward direction
* `y` — lateral position
* `z` — vertical position
* `h` — height
* `w` — width
* `l` — length
* `yaw` — rotation around the vertical axis

The eight corners are initially generated around the origin:

```text
              top
        0-------------1
       /               \
      3-----------------2

        4-------------5
       /               \
      7-----------------6
             bottom
```

The corners are then rotated around the Z-axis using the predicted yaw:

```python
rotation = np.array([
    [cos_yaw, -sin_yaw, 0],
    [sin_yaw,  cos_yaw, 0],
    [0,        0,       1]
])
```

Finally, the rotated corners are translated to the predicted object center.

---

# Visualization

The project includes an interactive **3D LiDAR visualization using Plotly**.

The visualization displays:

* LiDAR point cloud
* predicted 3D bounding boxes
* detection indices
* X/Y/Z coordinate axes

To keep visualization responsive, the point cloud is limited to at most:

```python
max_points = 30000
```

If more points are available, a random subset is selected.

```python
indices = np.random.choice(
    len(xyz),
    max_points,
    replace=False
)
```

---

## Coordinate System Visualization

The visualization explicitly displays the LiDAR coordinate axes:

```text
X → Forward
Y → Left / Right
Z → Up / Down
```

This makes it easier to understand the relationship between the LiDAR point cloud and predicted 3D bounding boxes.

---

# BEV Detection Visualization

The main inference result is visualized directly on the SFA3D BEV representation.

The project uses:

```python
bev_image = draw_predictions(
    bev_visual.copy(),
    detections[0].copy(),
    3
)
```

The resulting image is then rotated by 180 degrees:

```python
bev_image = cv2.rotate(
    bev_image,
    cv2.ROTATE_180
)
```

This follows the orientation used by the original SFA3D visualization pipeline.

The final visualization shows the predicted 3D detections from a top-down perspective.

---

# Acknowledgment

The implementation is based on the open-source **SFA3D** project and its associated codebase.

The SFA3D implementation provides the core model, BEV generation, decoding, post-processing, and coordinate-conversion components used in this project.

This repository contains my practical setup, inference workflow, and visualization work built around that implementation.

Original SFA3D project:

https://github.com/maudzung/SFA3D


