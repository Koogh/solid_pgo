# solid_pgo

SLAM backend for FAST_LIO: **SOLiD** loop detection + **GTSAM** pose graph optimization.

Based on [SC-PGO (gisbi-kim/FAST_LIO_SLAM)](https://github.com/gisbi-kim/FAST_LIO_SLAM) with ScanContext replaced by [SOLiD (sparolab/SOLiD)](https://github.com/sparolab/SOLiD).

Supports **ROS1 (noetic/melodic)** and **ROS2 (humble)**.

---

## Directory Structure

```
solid_pgo/
├── core/                          # ROS-free shared code
│   └── include/
│       └── solid_pgo/
│           ├── common.h           # PointType (PointXYZ), Pose6D
│           ├── SOLiDManager.h     # SOLiD loop closure wrapper
│           └── PoseGraphManager.h # GTSAM ISAM2 pose graph
│
├── ros1/                          # ROS1 noetic/melodic
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── src/solid_pgo_node.cpp
│   ├── config/params.yaml
│   └── launch/solid_pgo.launch
│
├── ros2/                          # ROS2 humble
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── src/solid_pgo_node.cpp
│   ├── config/params.yaml
│   └── launch/solid_pgo.launch.py
│
├── third_party/
│   └── SOLiD/                     # SOLiD source (pure C++, no ROS)
│       ├── include/solid_module.h
│       └── src/solid.cpp
│
└── README.md
```

**Design principle**: `core/` holds all ROS-independent logic. `ros1/` and `ros2/` only contain ROS API wrappers — ICP, GTSAM, and loop detection logic are identical between both.

---

## Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| PCL | ≥ 1.10 | Point cloud processing |
| GTSAM | ≥ 4.0 | Pose graph optimization |
| Eigen3 | ≥ 3.3 | Linear algebra |
| OpenMP | any | Parallel point cloud transforms |
| ROS1 noetic **or** ROS2 humble | — | One or both |

---

## Build

### ROS1 (catkin)

```bash
cd ~/catkin_ws/src
ln -s /path/to/solid_pgo/ros1 solid_pgo
cd ~/catkin_ws
catkin_make --only-pkg-with-deps solid_pgo
# or
catkin build solid_pgo
```

### ROS2 (colcon)

```bash
cd ~/ros2_ws/src
ln -s /path/to/solid_pgo/ros2 solid_pgo
cd ~/ros2_ws
colcon build --packages-select solid_pgo
source install/setup.bash
```

---

## Run

### ROS1

```bash
# Terminal 1: FAST_LIO
roslaunch fast_lio mapping_ouster64.launch

# Terminal 2: solid_pgo
roslaunch solid_pgo solid_pgo.launch
```

### ROS2

```bash
# Terminal 1: FAST_LIO ROS2
ros2 launch fast_lio mapping_ouster64.launch.py

# Terminal 2: solid_pgo
ros2 launch solid_pgo solid_pgo.launch.py
```

---

## Topics

### Subscribed

| Topic | Message Type | Source |
|-------|-------------|--------|
| `/velodyne_cloud_registered_local` (remapped from `/cloud_registered_body`) | PointCloud2 | FAST_LIO |
| `/aft_mapped_to_init` (remapped from `/Odometry`) | Odometry | FAST_LIO |
| `/gps/fix` | NavSatFix | GPS (optional) |

### Published

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/aft_pgo_odom` | Odometry | Latest optimized pose |
| `/aft_pgo_path` | Path | Full optimized trajectory |
| `/aft_pgo_map` | PointCloud2 | Optimized point cloud map |
| `/loop_scan_local` | PointCloud2 | Current scan in loop closure (debug) |
| `/loop_submap_local` | PointCloud2 | Historical submap (debug) |

---

## Parameters

### ROS1 (`config/params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keyframe_meter_gap` | 0.5 | [m] keyframe spacing |
| `keyframe_deg_gap` | 10.0 | [deg] keyframe rotation threshold |
| `solid_loop_threshold` | 0.7 | cosine similarity threshold (0~1, higher = stricter) |
| `solid_num_exclude_recent` | 50 | exclude latest N frames from loop search |
| `mapviz_filter_size` | 0.05 | [m] voxel size for map visualization |
| `save_directory` | `/tmp/solid_pgo_data/` | output directory (must end with `/`) |

### ROS2 (`config/params.yaml`)

Same parameters, under `solid_pgo_node: ros__parameters:` namespace.

---

## Tuning Guide

### `solid_loop_threshold` (cosine similarity)

SOLiD uses cosine similarity between range descriptors (higher = more similar):

| Value | Effect |
|-------|--------|
| 0.5 | Lenient — more loop candidates, more false positives |
| 0.7 | Recommended default for outdoor environments |
| 0.85+ | Strict — fewer false positives but may miss true loops |

ICP (fitness threshold 0.3) acts as a second verification stage, so you can start with a lenient threshold.

### `solid_num_exclude_recent`

Prevents detecting loops with very recent frames (which are not true revisits).
- 50 frames at 0.5 m/frame gap ≈ 25 m buffer

### `keyframe_meter_gap`

- Outdoor: 0.5 ~ 1.0 m
- Indoor: 0.3 ~ 0.5 m

---

## Changes vs SC-PGO

| Aspect | SC-PGO | solid_pgo |
|--------|--------|-----------|
| Loop detector | ScanContext (2D polar context) | SOLiD (1D range + angle descriptors) |
| Loop similarity metric | Distance (lower = better) | Cosine similarity (higher = better) |
| Yaw estimation | ScanContext sector shift | SOLiD angle descriptor circular shift |
| PointType | `pcl::PointXYZI` | `pcl::PointXYZ` |
| ROS2 support | No | Yes |
| Code structure | Single monolithic node | core/ + ros1/ + ros2/ |
| ICP | Identical | Identical |
| GTSAM / ISAM2 | Identical | Identical (in PoseGraphManager) |
| Topic names | Same | Same |

---

## FAST_LIO Integration

solid_pgo subscribes to:
- `/cloud_registered_body` (remapped to `/velodyne_cloud_registered_local`)
- `/Odometry` (remapped to `/aft_mapped_to_init`)

These are the default FAST_LIO output topics. The remapping is handled in the launch files.
