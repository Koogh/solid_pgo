# solid_pgo

**SLAM back-end for ROS 2 Humble** using [SOLiD](https://github.com/sparolab/solid) for loop detection and [GTSAM iSAM2](https://gtsam.org/) for pose graph optimization.

Designed to work with any LiDAR-Inertial Odometry (LIO) front-end that publishes `PointCloud2` + `nav_msgs/Odometry`.
Default topics are set to **rko_lio**, but all topics are configurable.

---

## Overview

```
LIO Front-end                       solid_pgo (Back-end)
─────────────────                   ─────────────────────────────────────────
rko_lio/frame      ──────────────►  Keyframe selection
rko_lio/odometry   ──────────────►  SOLiD descriptor  ──►  Loop detection (1 Hz)
                                    odom factor             │
                                    GTSAM iSAM2  ◄──────────┘  ICP verification
                                         │
                                         ▼
                              solid_pgo/path   (optimized trajectory)
                              solid_pgo/map    (global point cloud map)
                              solid_pgo/odometry
                              solid_pgo/loop_markers
```

### SOLiD Descriptor

Each LiDAR scan is encoded into a **100-dimensional vector** (40 range + 60 angle bins) via Range-Angle-Height (RAH) binning:

1. Points are projected into a polar grid of `(range, azimuth, elevation)` bins
2. A **height-distribution weight vector** is computed from the range matrix
3. `range_solid = range_matrix × weight` → encodes spatial density per range ring
4. `angle_solid = angle_matrix × weight` → encodes spatial density per azimuth sector

**Loop detection**: cosine similarity on the `range_solid` part
**Yaw initial guess for ICP**: L1-norm shift search on the `angle_solid` part

### Pose Graph Optimization

- **Sequential factors**: BetweenFactor from LIO odometry between consecutive keyframes
- **Loop factors**: BetweenFactor from ICP-verified loop closures (Cauchy robust kernel)
- **Solver**: GTSAM iSAM2 (incremental, runs after every new keyframe or loop closure)

---

## Architecture

Three threads run concurrently:

| Thread | Responsibility |
|--------|---------------|
| `process_pg` | Keyframe selection · odom factor insertion · iSAM2 update |
| `process_lcd` | SOLiD brute-force loop candidate search (configurable Hz) |
| `process_icp` | ICP verification with SOLID yaw hint · loop factor insertion |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| ROS 2 Humble | Middleware |
| GTSAM ≥ 4.1 | Pose graph optimization |
| PCL | Point cloud processing |
| `ros-humble-pcl-ros` | PCL ↔ ROS 2 bridge |
| `ros-humble-pcl-conversions` | PCL ↔ ROS 2 bridge |
| `ros-humble-visualization-msgs` | Loop closure markers |
| Eigen 3 | Linear algebra |

### Install GTSAM (Ubuntu 22.04)

```bash
sudo add-apt-repository ppa:borglab/gtsam-release-4.1 -y
sudo apt-get update
sudo apt-get install -y libgtsam-dev libgtsam-unstable-dev
```

### Install ROS 2 packages

```bash
sudo apt-get install -y \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-visualization-msgs
```

---

## Build

```bash
cd ~/your_ros2_ws
colcon build --packages-select solid_pgo --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

---

## Usage

### Launch with default topics (rko_lio)

```bash
ros2 launch solid_pgo solid_pgo.launch.py
```

### Launch with custom topics

```bash
ros2 launch solid_pgo solid_pgo.launch.py \
    cloud_topic:=/your/cloud \
    odom_topic:=/your/odometry \
    map_frame:=map
```

### Run directly

```bash
ros2 run solid_pgo solid_pgo_node \
    --ros-args --params-file path/to/params.yaml
```

---

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `rko_lio/frame` *(configurable)* | `sensor_msgs/PointCloud2` | Deskewed LiDAR scan from LIO |
| `rko_lio/odometry` *(configurable)* | `nav_msgs/Odometry` | LIO pose estimate |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `solid_pgo/path` | `nav_msgs/Path` | Full optimized trajectory |
| `solid_pgo/odometry` | `nav_msgs/Odometry` | Latest optimized pose |
| `solid_pgo/map` | `sensor_msgs/PointCloud2` | Global voxel-downsampled map |
| `solid_pgo/loop_markers` | `visualization_msgs/MarkerArray` | Accepted loop closure edges |
| `solid_pgo/loop_scan` | `sensor_msgs/PointCloud2` | Current scan at loop (debug) |
| `solid_pgo/loop_submap` | `sensor_msgs/PointCloud2` | Loop candidate submap (debug) |

### TF

Publishes `map → base_link` transform after each iSAM2 update.

---

## Configuration

All parameters are set in [`config/params.yaml`](config/params.yaml).

### LiDAR vertical FOV — tune to your sensor

| Sensor | `solid_fov_upper` | `solid_fov_lower` | `solid_num_height` |
|--------|:-----------------:|:-----------------:|:------------------:|
| VLP-16 | 15.0 | -15.0 | 16 |
| OS1-64 | 22.5 | -22.5 | 64 |
| Avia | 20.6 | -51.4 | 6 |
| *Default (LOAM-like)* | 2.0 | -24.8 | 32 |

### Key parameters

```yaml
# Keyframe selection
keyframe_meter_gap: 1.0   # [m]   new keyframe every N meters
keyframe_rad_gap:   0.4   # [rad] new keyframe every N radians

# SOLiD loop detection
solid_loop_threshold:     0.85  # cosine similarity — raise to reduce false positives
solid_num_exclude_recent: 50    # skip last N keyframes (avoid near-self matches)
loop_closure_frequency:   1.0   # [Hz]

# ICP verification
loop_submap_size:      10   # ±N keyframes around loop candidate
icp_fitness_threshold: 0.3  # reject if ICP score > this
```

### Odometry noise mode

Controls how uncertainty is assigned to each odometry factor in the pose graph.

| `odom_noise_mode` | Description |
|-------------------|-------------|
| `"fixed"` | Constant diagonal noise *(default, always works)* |
| `"distance"` | Noise scales linearly with distance traveled |
| `"covariance"` | Uses `pose.covariance` from the odometry message; falls back to `"fixed"` if covariance is all-zero |

Switch modes with a single parameter change — no code change needed:

```yaml
odom_noise_mode: "distance"   # or "fixed" / "covariance"
```

---

## File Structure

```
solid_pgo/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── params.yaml              # all tunable parameters
├── launch/
│   └── solid_pgo.launch.py
├── include/solid_pgo/
│   ├── solid_module.hpp         # SOLiD descriptor (header-only, ported from sparolab/solid)
│   └── odom_noise.hpp           # Pluggable odometry noise model
└── src/
    └── solid_pgo_node.cpp       # Main ROS 2 node
```

---

## References

- **SOLiD descriptor**
  J.-H. Jung et al., *"SOLiD: Spatially Organized and Lightweight Global Descriptor for FOV-constrained LiDAR Place Recognition"*, RA-L 2024.
  [https://github.com/sparolab/solid](https://github.com/sparolab/solid)

- **SC-PGO** (architecture reference)
  G. Kim and A. Kim, *"Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D Point Cloud Map"*, IROS 2018.
  [https://github.com/gisbi-kim/SC-PGO](https://github.com/gisbi-kim/SC-PGO)

- **GTSAM**
  [https://gtsam.org](https://gtsam.org)
