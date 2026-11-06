#ifndef LIDAR_OBJECT_DETECTION__SCAN_SEAM_HPP_
#define LIDAR_OBJECT_DETECTION__SCAN_SEAM_HPP_

#include <cmath>
#include <utility>
#include <vector>

#include "lidar_object_detection/tracked_object.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

namespace lidar_object_detection
{

// True when the scan's beams wrap all the way round, i.e. its last beam is adjacent to its
// first. Only then do the first and last clusters of a scan border each other.
inline bool scan_covers_full_circle(const sensor_msgs::msg::LaserScan & scan)
{
  const double span = static_cast<double>(scan.angle_max - scan.angle_min) +
                      std::abs(static_cast<double>(scan.angle_increment));
  return span >= 2.0 * M_PI - 1e-3;
}

// In a full-circle scan an object straddling the first/last beam is split into the last
// and the first cluster, and each half can fail the point-count or width filters although
// the whole object passes. Rejoin them (last + first keeps the points in angular order)
// when their facing endpoints are closer than the clustering gap.
inline void merge_seam_clusters(
  std::vector<std::vector<Point2D>> & clusters, double max_dist_between_points)
{
  if (clusters.size() < 2 || clusters.front().empty() || clusters.back().empty()) {
    return;
  }
  const Point2D & tail = clusters.back().back();
  const Point2D & head = clusters.front().front();
  if (std::hypot(head.x - tail.x, head.y - tail.y) >= max_dist_between_points) {
    return;
  }
  std::vector<Point2D> joined = std::move(clusters.back());
  joined.insert(joined.end(), clusters.front().begin(), clusters.front().end());
  clusters.front() = std::move(joined);
  clusters.pop_back();
}

}  // namespace lidar_object_detection

#endif  // LIDAR_OBJECT_DETECTION__SCAN_SEAM_HPP_
