// Unit tests for the tracker's velocity estimate / miss handling and the full-circle
// scan-seam cluster merge. Pure logic: no executor, no topics.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "lidar_object_detection/scan_seam.hpp"
#include "lidar_object_detection/tracked_object.hpp"

namespace
{

ObjectCandidate at(double x, double y)
{
  ObjectCandidate c;
  c.center = {x, y};
  c.points = {{x - 0.05, y}, {x, y + 0.05}, {x + 0.05, y}};
  c.closest_distance = std::hypot(x, y);
  return c;
}

rclcpp::Time t(double s) { return rclcpp::Time(static_cast<int64_t>(s * 1e9)); }

// The reviewer's case: seen at -0.2 moving at -2 m/s, coasted twice to -0.6, re-seen at
// -0.5 (it slowed while unseen). Differencing against the coasted -0.6 over one interval
// would report +1 m/s (wrong sign); across the observations it is -1 m/s.
TEST(TrackedObject, VelocityIsMeasuredBetweenObservationsNotFromThePrediction)
{
  TrackedObject track(1, at(0.0, 1.0), t(0.0));
  track.update(at(-0.2, 1.0), t(0.1));
  EXPECT_NEAR(track.velocity_.x, -2.0, 1e-9);

  track.mark_missed(t(0.2), /*predict=*/true);
  track.mark_missed(t(0.3), /*predict=*/true);
  EXPECT_NEAR(track.center_.x, -0.6, 1e-9);

  track.update(at(-0.5, 1.0), t(0.4));
  EXPECT_NEAR(track.velocity_.x, (-0.5 - -0.2) / 0.3, 1e-9);
  EXPECT_LT(track.velocity_.x, 0.0);
  EXPECT_EQ(track.staleness_, 0);
}

TEST(TrackedObject, MissWithoutPredictionHoldsTheLastPosition)
{
  TrackedObject track(1, at(0.0, 1.0), t(0.0));
  track.update(at(-0.2, 1.0), t(0.1));
  track.mark_missed(t(0.2), /*predict=*/false);
  EXPECT_NEAR(track.center_.x, -0.2, 1e-9);
  EXPECT_EQ(track.staleness_, 1);
}

TEST(TrackedObject, MissBeforeAVelocityExistsDoesNotMove)
{
  TrackedObject track(1, at(0.3, 1.0), t(0.0));
  track.mark_missed(t(0.1), /*predict=*/true);
  EXPECT_FALSE(track.has_velocity_);
  EXPECT_NEAR(track.center_.x, 0.3, 1e-9);
}

sensor_msgs::msg::LaserScan scan_span(double angle_min, double angle_max, double inc)
{
  sensor_msgs::msg::LaserScan s;
  s.angle_min = static_cast<float>(angle_min);
  s.angle_max = static_cast<float>(angle_max);
  s.angle_increment = static_cast<float>(inc);
  return s;
}

TEST(ScanSeam, FullCircleDetection)
{
  const double inc = 2.0 * M_PI / 720.0;
  EXPECT_TRUE(lidar_object_detection::scan_covers_full_circle(scan_span(-M_PI, M_PI - inc, inc)));
  EXPECT_FALSE(lidar_object_detection::scan_covers_full_circle(scan_span(-2.0, 2.0, inc)));
}

TEST(ScanSeam, ObjectAcrossTheSeamIsRejoined)
{
  // Behind the sensor: last cluster ends just before +pi, first starts just after -pi.
  std::vector<std::vector<Point2D>> clusters = {
    {{-2.0, -0.02}, {-2.0, -0.06}},  // first cluster (angles just above -pi)
    {{0.0, 2.0}, {0.1, 2.0}},        // unrelated object
    {{-2.0, 0.06}, {-2.0, 0.02}},    // last cluster (angles just below +pi)
  };
  lidar_object_detection::merge_seam_clusters(clusters, 0.3);
  ASSERT_EQ(clusters.size(), 2u);
  ASSERT_EQ(clusters.front().size(), 4u);
  EXPECT_DOUBLE_EQ(clusters.front().front().y, 0.06);  // last cluster's points come first
  EXPECT_DOUBLE_EQ(clusters.front().back().y, -0.06);
}

TEST(ScanSeam, DistantEndsStaySeparate)
{
  std::vector<std::vector<Point2D>> clusters = {{{-2.0, -0.5}}, {{0.0, 2.0}}, {{-2.0, 0.5}}};
  lidar_object_detection::merge_seam_clusters(clusters, 0.3);
  EXPECT_EQ(clusters.size(), 3u);
}

}  // namespace
