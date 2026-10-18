#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>

#include <lidar_object_detection/msg/detected_object_array.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

}  // namespace

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
