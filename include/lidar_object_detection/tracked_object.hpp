#ifndef LIDAR_OBJECT_DETECTION__TRACKED_OBJECT_HPP_
#define LIDAR_OBJECT_DETECTION__TRACKED_OBJECT_HPP_

#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/time.hpp"

// A simple struct to represent a point in 2D space
struct Point2D
{
  double x, y;
};

// Represents a cluster of points that could be an object
struct ObjectCandidate
{
  std::vector<Point2D> points;
  Point2D center;
  double closest_distance;
};

// Represents an object that is being tracked across frames
class TrackedObject
{
public:
  TrackedObject(uint32_t id, const ObjectCandidate & candidate, const rclcpp::Time & time)
  : object_id_(id),
    center_(candidate.center),
    last_update_time_(time),
    staleness_(0),
    observed_center_(candidate.center),
    center_time_(time)
  {
    velocity_.x = 0;
    velocity_.y = 0;
    velocity_.z = 0;  // 2D, so z is 0

    // Find the convex hull for the bounding box
    bounding_box_ = calculate_convex_hull(candidate.points);
  }

  void update(const ObjectCandidate & candidate, const rclcpp::Time & time)
  {
    // Difference against the last *observation*, over the whole interval since it, never
    // against a coasted center_: that is a prediction, and its error divided by one scan
    // interval can even flip the sign of the estimate when the object changed speed while
    // unseen.
    const double elapsed_s = (time - last_update_time_).seconds();
    if (elapsed_s > 1e-6) {  // Avoid division by zero
      velocity_.x = (candidate.center.x - observed_center_.x) / elapsed_s;
      velocity_.y = (candidate.center.y - observed_center_.y) / elapsed_s;
      has_velocity_ = true;
    }

    observed_center_ = candidate.center;
    center_ = candidate.center;
    bounding_box_ = calculate_convex_hull(candidate.points);
    last_update_time_ = time;
    center_time_ = time;
    staleness_ = 0;
  }

  // Unmatched this scan: keep the track for re-association. With `predict` it coasts at
  // its last (sensor-frame, i.e. sensor-relative) velocity — frozen in the sensor frame a
  // track rides along with a moving sensor and looks like a moving object to every
  // consumer; coasted, a static wall stays static in the world. Without `predict` it holds
  // its last position (the behaviour for a static sensor, or when consumers prefer it).
  void mark_missed(const rclcpp::Time & time, bool predict)
  {
    staleness_++;
    if (!predict || !has_velocity_) return;
    const double dt = (time - center_time_).seconds();
    if (dt <= 0.0) return;
    const double dx = velocity_.x * dt;
    const double dy = velocity_.y * dt;
    center_.x += dx;
    center_.y += dy;
    for (auto & p : bounding_box_) {
      p.x += static_cast<float>(dx);
      p.y += static_cast<float>(dy);
    }
    center_time_ = time;
  }

  // Public member variables for easy access
  uint32_t object_id_;
  Point2D center_;
  geometry_msgs::msg::Vector3 velocity_;
  std::vector<geometry_msgs::msg::Point32> bounding_box_;
  rclcpp::Time last_update_time_;
  int staleness_;
  // Seen at least twice: velocity_ is an estimate (a coasting track without one is not
  // published, its position is unknown).
  bool has_velocity_{false};

private:
  Point2D observed_center_;   // the last measured center (center_ may be a prediction)
  rclcpp::Time center_time_;  // the time center_ refers to (last sighting or coast step)

  // Calculates the convex hull of a set of 2D points using the Monotone Chain algorithm.
  // This is used to create a tight bounding polygon for visualization and analysis.
  std::vector<geometry_msgs::msg::Point32> calculate_convex_hull(
    const std::vector<Point2D> & points)
  {
    std::vector<geometry_msgs::msg::Point32> hull;
    if (points.size() < 3) {
      for (const auto & p : points) {
        geometry_msgs::msg::Point32 p32;
        p32.x = p.x;
        p32.y = p.y;
        p32.z = 0.0;
        hull.push_back(p32);
      }
      return hull;
    }

    auto cross_product = [](Point2D a, Point2D b, Point2D c) {
      return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };

    std::vector<Point2D> sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end(), [](Point2D a, Point2D b) {
      return a.x < b.x || (a.x == b.x && a.y < b.y);
    });

    // Lower hull
    for (const auto & p : sorted_points) {
      while (hull.size() >= 2 && cross_product(
                                   {hull[hull.size() - 2].x, hull[hull.size() - 2].y},
                                   {hull.back().x, hull.back().y}, p) <= 0) {
        hull.pop_back();
      }
      geometry_msgs::msg::Point32 p32;
      p32.x = p.x;
      p32.y = p.y;
      p32.z = 0.0;
      hull.push_back(p32);
    }

    // Upper hull
    size_t lower_hull_size = hull.size();
    for (int i = sorted_points.size() - 2; i >= 0; --i) {
      const auto & p = sorted_points[i];
      while (hull.size() > lower_hull_size && cross_product(
                                                {hull[hull.size() - 2].x, hull[hull.size() - 2].y},
                                                {hull.back().x, hull.back().y}, p) <= 0) {
        hull.pop_back();
      }
      geometry_msgs::msg::Point32 p32;
      p32.x = p.x;
      p32.y = p.y;
      p32.z = 0.0;
      hull.push_back(p32);
    }
    hull.pop_back();  // The last point is the same as the first
    return hull;
  }
};

#endif  // LIDAR_OBJECT_DETECTION__TRACKED_OBJECT_HPP_