#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>

#include <lidar_object_detection/msg/detected_object_array.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lidar_object_detection/lidar_object_detection_node.hpp"

namespace
{

// Parameter overrides that satisfy every required (no-default) parameter with
// marker toggles ON. A synthetic cluster sits at 2 m inside a 30-degree FOV.
std::vector<rclcpp::Parameter> base_params(bool markers_on)
{
  return {
    rclcpp::Parameter("max_object_width_m", 1.0),
    rclcpp::Parameter("min_object_width_m", 0.1),
    rclcpp::Parameter("min_amount_of_object_points", 5),
    rclcpp::Parameter("detection_angle_deg", 30.0),
    rclcpp::Parameter("marker_lifetime_cycles", 5.0),
    rclcpp::Parameter("markers.objects", markers_on),
    rclcpp::Parameter("markers.fov", markers_on),
    rclcpp::Parameter("markers.tracking_zone", markers_on),
  };
}

sensor_msgs::msg::LaserScan cluster_scan(int seq)
{
  sensor_msgs::msg::LaserScan scan;
  scan.header.frame_id = "laser";
  scan.header.stamp = rclcpp::Time(seq * 100000000LL);  // 0.1 s apart
  scan.angle_min = -1.0;
  scan.angle_max = 1.0;
  scan.angle_increment = 0.01;
  scan.time_increment = 0.0;
  scan.scan_time = 0.1;
  scan.range_min = 0.1;
  scan.range_max = 12.0;
  scan.ranges.assign(
    static_cast<size_t>((scan.angle_max - scan.angle_min) / scan.angle_increment),
    std::numeric_limits<float>::quiet_NaN());
  // 20 returns at ~2 m spanning 0.19 rad (~0.38 m chord) inside the FOV.
  for (int i = 0; i < 20; ++i) {
    scan.ranges[static_cast<size_t>(90 + i)] = 2.0f;
  }
  return scan;
}

// Parameter overrides for the tracking/coasting tests: markers off (the toggles
// are required parameters but not interesting here), with the detection arc,
// point floor and coasting switch under test.
std::vector<rclcpp::Parameter> tracking_params(
  double detection_angle_deg, int min_points, bool coast)
{
  return {
    rclcpp::Parameter("max_object_width_m", 1.0),
    rclcpp::Parameter("min_object_width_m", 0.1),
    rclcpp::Parameter("min_amount_of_object_points", min_points),
    rclcpp::Parameter("detection_angle_deg", detection_angle_deg),
    rclcpp::Parameter("marker_lifetime_cycles", 5.0),
    rclcpp::Parameter("markers.objects", false),
    rclcpp::Parameter("markers.fov", false),
    rclcpp::Parameter("markers.tracking_zone", false),
    rclcpp::Parameter("tracking_coast_unmatched", coast),
  };
}

// cluster_scan's geometry with no returns: one cycle in which the established
// track is unmatched.
sensor_msgs::msg::LaserScan blank_scan(int seq)
{
  auto scan = cluster_scan(seq);
  std::fill(
    scan.ranges.begin(), scan.ranges.end(), std::numeric_limits<float>::quiet_NaN());
  return scan;
}

// Full-circle scan whose first beam sits at float(-pi) -- the case where
// float(-pi) is ~9e-8 below the double -M_PI, so a plain range test drops it.
sensor_msgs::msg::LaserScan full_circle_scan(int seq, const std::vector<int> & beams)
{
  sensor_msgs::msg::LaserScan scan;
  scan.header.frame_id = "laser";
  scan.header.stamp = rclcpp::Time(seq * 100000000LL);  // 0.1 s apart
  constexpr int beam_count = 360;
  const double increment = 2.0 * M_PI / beam_count;
  scan.angle_min = static_cast<float>(-M_PI);  // float(-pi), deliberately not a double
  scan.angle_increment = static_cast<float>(increment);
  scan.angle_max = static_cast<float>(-M_PI + (beam_count - 1) * increment);
  scan.time_increment = 0.0f;
  scan.scan_time = 0.1f;
  scan.range_min = 0.1f;
  scan.range_max = 12.0f;
  scan.ranges.assign(beam_count, std::numeric_limits<float>::quiet_NaN());
  for (const int i : beams) {
    scan.ranges[static_cast<size_t>(i)] = 2.0f;
  }
  return scan;
}

// Object straddling the seam: beams 176..179 deg plus -180..-176 deg, 9 returns
// in total, so losing the first beam leaves 8 and fails a floor of 9.
sensor_msgs::msg::LaserScan seam_scan(int seq)
{
  return full_circle_scan(seq, {0, 1, 2, 3, 4, 356, 357, 358, 359});
}

// Harness: detection node + a scan publisher, spun together. Collects the
// messages each watched topic receives during `scans` scan cycles.
struct Harness
{
  rclcpp::Node::SharedPtr driver;
  std::shared_ptr<LidarObjectDetectionNode> detection;
  rclcpp::executors::SingleThreadedExecutor executor;

  std::vector<lidar_object_detection::msg::DetectedObjectArray> detected;
  std::vector<visualization_msgs::msg::MarkerArray> object_markers;
  std::vector<visualization_msgs::msg::Marker> fov_markers;
  std::vector<visualization_msgs::msg::Marker> zone_markers;

  explicit Harness(bool markers_on) : driver(rclcpp::Node::make_shared("scan_driver"))
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(base_params(markers_on));
    detection = std::make_shared<LidarObjectDetectionNode>(options);

    auto scan_pub = driver->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
    // Keep-alive: subscriptions must outlive the collection loop below.
    auto detected_sub =
      driver->create_subscription<lidar_object_detection::msg::DetectedObjectArray>(
        "detected", 10, [this](lidar_object_detection::msg::DetectedObjectArray::SharedPtr msg) {
          detected.push_back(*msg);
        });
    auto markers_sub = driver->create_subscription<visualization_msgs::msg::MarkerArray>(
      "markers", 10, [this](visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        object_markers.push_back(*msg);
      });
    auto fov_sub = driver->create_subscription<visualization_msgs::msg::Marker>(
      "fov_marker", 10,
      [this](visualization_msgs::msg::Marker::SharedPtr msg) { fov_markers.push_back(*msg); });
    auto zone_sub = driver->create_subscription<visualization_msgs::msg::Marker>(
      "tracking_zone_marker", 10,
      [this](visualization_msgs::msg::Marker::SharedPtr msg) { zone_markers.push_back(*msg); });
    (void)detected_sub;
    (void)markers_sub;
    (void)fov_sub;
    (void)zone_sub;

    executor.add_node(driver);
    executor.add_node(detection);
    for (int seq = 0; seq < scans; ++seq) {
      scan_pub->publish(cluster_scan(seq));
      // Give the executor a moment per cycle.
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
      while (std::chrono::steady_clock::now() < deadline) {
        executor.spin_once(std::chrono::milliseconds(5));
      }
    }
  }

  static constexpr int scans = 10;
};

// The marker toggles are required (no defaults): with every other parameter
// provided, omitting them must fail node startup.
TEST(LidarObjectDetectionMarkers, ThrowsWhenMarkerTogglesMissing)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(
    {rclcpp::Parameter("max_object_width_m", 1.0), rclcpp::Parameter("min_object_width_m", 0.1),
     rclcpp::Parameter("min_amount_of_object_points", 5),
     rclcpp::Parameter("detection_angle_deg", 30.0),
     rclcpp::Parameter("marker_lifetime_cycles", 5.0)});
  EXPECT_THROW(
    LidarObjectDetectionNode node(options),
    rclcpp::exceptions::UninitializedStaticallyTypedParameterException);
}

// An invalid startup configuration must fail node construction rather than
// leave a node running that silently detects nothing.
TEST(LidarObjectDetectionParams, ThrowsOnInvalidStartupConfig)
{
  auto params = base_params(true);
  params.emplace_back("detection_angle_deg", 0.0);  // later override wins
  rclcpp::NodeOptions options;
  options.parameter_overrides(params);
  EXPECT_THROW(LidarObjectDetectionNode node(options), std::invalid_argument);
}

// A runtime set of an invalid value is rejected with a reason and never
// applied; a valid one is accepted.
TEST(LidarObjectDetectionParams, RejectsInvalidRuntimeValues)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(base_params(true));
  auto node = std::make_shared<LidarObjectDetectionNode>(options);

  auto bad_angle = node->set_parameter(rclcpp::Parameter("detection_angle_deg", 0.0));
  EXPECT_FALSE(bad_angle.successful);
  EXPECT_FALSE(bad_angle.reason.empty());
  EXPECT_DOUBLE_EQ(node->get_parameter("detection_angle_deg").as_double(), 30.0);

  // Invalid only in combination with the current max width (1.0).
  auto bad_min = node->set_parameter(rclcpp::Parameter("min_object_width_m", 2.0));
  EXPECT_FALSE(bad_min.successful);
  EXPECT_DOUBLE_EQ(node->get_parameter("min_object_width_m").as_double(), 0.1);

  // Valid together even though min >= the *current* max: checked as a set.
  auto both = node->set_parameters_atomically(
    {rclcpp::Parameter("max_object_width_m", 3.0), rclcpp::Parameter("min_object_width_m", 2.0)});
  EXPECT_TRUE(both.successful) << both.reason;

  auto ok = node->set_parameter(rclcpp::Parameter("detection_angle_deg", 45.0));
  EXPECT_TRUE(ok.successful) << ok.reason;
}

// markers.*=false must suppress every marker stream while detection keeps
// publishing real tracked objects — the gate may never touch driving data.
TEST(LidarObjectDetectionMarkers, DisabledTogglesKeepDetectionAndDropMarkers)
{
  Harness h(/*markers_on=*/false);
  ASSERT_FALSE(h.detected.empty());
  bool saw_object = false;
  for (const auto & msg : h.detected) {
    if (!msg.objects.empty()) {
      saw_object = true;
    }
  }
  EXPECT_TRUE(saw_object) << "detection must keep working with markers off";
  EXPECT_TRUE(h.object_markers.empty());
  EXPECT_TRUE(h.fov_markers.empty());
  EXPECT_TRUE(h.zone_markers.empty());
}

// markers.*=true keeps all three marker streams flowing.
TEST(LidarObjectDetectionMarkers, EnabledTogglesPublishMarkers)
{
  Harness h(/*markers_on=*/true);
  ASSERT_FALSE(h.detected.empty());
  EXPECT_FALSE(h.object_markers.empty());
  EXPECT_FALSE(h.fov_markers.empty());
  EXPECT_FALSE(h.zone_markers.empty());
}

// Drives one scan cycle at a time so a test can inspect what a single publish
// produced; the Harness above only exposes the whole run.
struct CycleHarness
{
  rclcpp::Node::SharedPtr driver;
  std::shared_ptr<LidarObjectDetectionNode> detection;
  rclcpp::executors::SingleThreadedExecutor executor;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub;
  rclcpp::Subscription<lidar_object_detection::msg::DetectedObjectArray>::SharedPtr detected_sub;
  std::vector<lidar_object_detection::msg::DetectedObjectArray> detected;

  explicit CycleHarness(std::vector<rclcpp::Parameter> overrides)
  : driver(rclcpp::Node::make_shared("scan_driver_cycle"))
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(std::move(overrides));
    detection = std::make_shared<LidarObjectDetectionNode>(options);

    scan_pub = driver->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
    detected_sub = driver->create_subscription<lidar_object_detection::msg::DetectedObjectArray>(
      "detected", 10, [this](lidar_object_detection::msg::DetectedObjectArray::SharedPtr msg) {
        detected.push_back(*msg);
      });

    executor.add_node(driver);
    executor.add_node(detection);
  }

  // Publish one scan, spin until that cycle's `detected` message lands, and
  // return the objects it carried. An empty vector means none were published
  // (either suppressed by the coasting filter or nothing was detected).
  std::vector<lidar_object_detection::msg::DetectedObject> cycle(
    const sensor_msgs::msg::LaserScan & scan)
  {
    const std::size_t before = detected.size();
    scan_pub->publish(scan);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (detected.size() == before && std::chrono::steady_clock::now() < deadline) {
      executor.spin_once(std::chrono::milliseconds(10));
    }
    if (detected.size() == before) {
      ADD_FAILURE() << "no `detected` message for the published scan";
      return {};
    }
    return detected.back().objects;
  }
};

// A full-circle arc must cluster the beam at float(-pi) too; dropping it leaves
// a seam-straddling object one point short of min_amount_of_object_points.
TEST(LidarObjectDetectionFullCircle, SeamObjectKeepsTheFirstBeam)
{
  CycleHarness h(tracking_params(/*detection_angle_deg=*/360.0, /*min_points=*/9, /*coast=*/true));
  const auto objects = h.cycle(seam_scan(0));
  ASSERT_EQ(objects.size(), 1u)
    << "object straddling the scan seam was not detected: the first beam at float(-pi) "
       "must not be range-filtered out of a 360 degree clustering arc";
}

// The same dropped beam seen without a seam straddle: the object is only the
// first five beams, so losing the one at float(-pi) puts it under the floor.
TEST(LidarObjectDetectionFullCircle, FirstBeamsObjectIsDetected)
{
  CycleHarness h(tracking_params(/*detection_angle_deg=*/360.0, /*min_points=*/5, /*coast=*/true));
  const auto objects = h.cycle(full_circle_scan(0, {0, 1, 2, 3, 4}));
  ASSERT_EQ(objects.size(), 1u)
    << "an object made of the scan's first beams must not be dropped by the "
       "float(-pi) range test";
}

// Seen once and then unmatched: there is no velocity to coast with, so the
// predicted position would be a guess -- it must not be published.
TEST(LidarObjectDetectionCoasting, SuppressesATrackThatHasNoVelocityYet)
{
  CycleHarness h(tracking_params(/*detection_angle_deg=*/30.0, /*min_points=*/5, /*coast=*/true));
  ASSERT_EQ(h.cycle(cluster_scan(0)).size(), 1u);
  EXPECT_TRUE(h.cycle(blank_scan(1)).empty());
}

// Seen twice (so a velocity exists) and then unmatched: published, at its
// coasted position.
TEST(LidarObjectDetectionCoasting, PublishesACoastingTrackWithAVelocityEstimate)
{
  CycleHarness h(tracking_params(/*detection_angle_deg=*/30.0, /*min_points=*/5, /*coast=*/true));
  ASSERT_EQ(h.cycle(cluster_scan(0)).size(), 1u);
  ASSERT_EQ(h.cycle(cluster_scan(1)).size(), 1u);
  EXPECT_EQ(h.cycle(blank_scan(2)).size(), 1u);
}

// Coasting disabled: an unmatched track is published where it was last seen.
TEST(LidarObjectDetectionCoasting, HoldsLastPositionWhenCoastingIsDisabled)
{
  CycleHarness h(tracking_params(/*detection_angle_deg=*/30.0, /*min_points=*/5, /*coast=*/false));
  const auto seen = h.cycle(cluster_scan(0));
  ASSERT_EQ(seen.size(), 1u);

  const auto missed = h.cycle(blank_scan(1));
  ASSERT_EQ(missed.size(), 1u) << "coasting off must keep publishing the track";
  EXPECT_NEAR(missed[0].center.x, seen[0].center.x, 1e-6);
  EXPECT_NEAR(missed[0].center.y, seen[0].center.y, 1e-6);
}

}  // namespace

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
