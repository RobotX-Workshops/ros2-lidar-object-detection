# 2D Lidar Object Detection

Lightweight 2D lidar-based object detection utilities.

A ROS2 package that implements simple 2D LIDAR object detection algorithms and helper scripts. It is intended to be extracted and maintained as a standalone project or included as a git submodule in larger workspaces.

## Contents

- ROS2 package source
- Config and launch examples

## Quick start

Requirements:
- ROS2 (compatible distribution used by the parent project)
- Any system dependencies listed in the package manifest (package.xml / setup files)

To use inside the parent workspace (no submodule):

1. Place this folder at `src/lidar_object_detection` inside a ROS2 workspace.
2. Build with colcon:

   ```bash
   colcon build --packages-select lidar_object_detection
   ```

## Parameters

Required parameters have no default: the node refuses to start without them, so every
deployment states its sensor geometry explicitly. Everything is runtime-reconfigurable
(`ros2 param set`); invalid values are rejected.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `max_object_width_m` | double | required | Widest cluster accepted as an object; also the max gap between neighbouring points of one cluster |
| `min_object_width_m` | double | required | Narrowest cluster accepted as an object |
| `min_amount_of_object_points` | int | required | Fewest returns in a cluster |
| `detection_angle_deg` | double | required | Field of view (centred on +x) in which objects are published. Clustering and tracking run on a wider arc so objects at the edge are not cut; at 360° a cluster crossing the scan's first/last beam is joined back together |
| `marker_lifetime_cycles` | double | required | Marker lifetime, in measured scan cycles |
| `markers.objects`, `markers.fov`, `markers.tracking_zone` | bool | required | Visualization toggles; off skips the marker computation, detection is unaffected |
| `tracking_dist_threshold_m` | double | `0.3` | Max distance between a track and a detection for them to be matched |
| `tracking_staleness_threshold` | int | `5` | Scans a track survives without a match |
| `tracking_coast_unmatched` | bool | `true` | Unmatched tracks coast at their last (sensor-relative) velocity; `false` holds them at their last position. Coasting suits a sensor on a moving platform — a held track would ride along with the sensor and look like a moving object. With coasting, a track seen only once (no velocity yet) is not published while unmatched |

Velocities are estimated between consecutive *observations* of a track (never from a
coasted prediction), in the sensor frame.

## Making this a standalone git repository

See the parent repository documentation or the instructions below for two common approaches:

- Simple export (no history)
- Preserve history using `git subtree split` (recommended if you want to keep commit history)

## License

This project is released under the MIT License — see `LICENSE`.

## Contributing

If you intend to contribute changes back to the original monorepo, prefer opening pull requests against whichever upstream repository is used to host this package.
