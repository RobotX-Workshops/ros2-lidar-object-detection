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

## Making this a standalone git repository

See the parent repository documentation or the instructions below for two common approaches:

- Simple export (no history)
- Preserve history using `git subtree split` (recommended if you want to keep commit history)

## License

This project is released under the MIT License — see `LICENSE`.

## Contributing

If you intend to contribute changes back to the original monorepo, prefer opening pull requests against whichever upstream repository is used to host this package.
