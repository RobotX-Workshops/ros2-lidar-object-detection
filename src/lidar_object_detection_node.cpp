#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <sstream>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "tf2/utils.h"
#include "lidar_object_detection/tracked_object.hpp"
#include "lidar_object_detection/msg/detected_object.hpp"
#include "lidar_object_detection/msg/detected_object_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std::chrono_literals;

class LidarObjectDetectionNode : public rclcpp::Node
{
public:
    LidarObjectDetectionNode() : Node("lidar_object_detection_node", rclcpp::NodeOptions().enable_logger_service(true)), next_object_id_(0)
    {
        // --- Parameters ---
        // Hardware-specific parameters (no defaults - must be configured)
        this->declare_parameter<double>(PARAM_MAX_OBJECT_WIDTH);
        this->declare_parameter<double>(PARAM_MIN_OBJECT_WIDTH);
        this->declare_parameter<int>(PARAM_MIN_OBJECT_POINTS);
        this->declare_parameter<double>(PARAM_DETECTION_ANGLE);

        // Behavioral parameters (with defaults)
        this->declare_parameter<double>(PARAM_TRACKING_DIST_THRESHOLD, 0.3);
        this->declare_parameter<int>(PARAM_TRACKING_STALENESS_THRESHOLD, 5);

        // Set up parameter callback for runtime updates
        param_callback_handle_ = this->add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter> &params) -> rcl_interfaces::msg::SetParametersResult
            {
                rcl_interfaces::msg::SetParametersResult result;
                result.successful = true;

                // Schedule delayed update to ensure parameters are set
                this->parameter_update_timer_ = this->create_wall_timer(
                    std::chrono::milliseconds(100),
                    [this]()
                    {
                        this->update_config();
                        this->parameter_update_timer_->cancel();
                    });

                return result;
            });

        // Initial parameter validation and configuration
        update_config();

        // --- Publishers ---
        objects_pub_ = this->create_publisher<lidar_object_detection::msg::DetectedObjectArray>("detected_objects", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("objects/visualization_markers", 10);
        fov_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("objects/fov_marker", 10);

        // --- Subscriber ---
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10, std::bind(&LidarObjectDetectionNode::scan_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Lidar Object Detection Node has been started.");
    }

private:
    // --- Parameter Constants ---
    static const std::string PARAM_MAX_OBJECT_WIDTH;
    static const std::string PARAM_MIN_OBJECT_WIDTH;
    static const std::string PARAM_MIN_OBJECT_POINTS;
    static const std::string PARAM_DETECTION_ANGLE;
    static const std::string PARAM_TRACKING_DIST_THRESHOLD;
    static const std::string PARAM_TRACKING_STALENESS_THRESHOLD;

    // --- Member Variables ---
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<lidar_object_detection::msg::DetectedObjectArray>::SharedPtr objects_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr fov_marker_pub_;

    // Parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // Timer for delayed parameter updates
    rclcpp::TimerBase::SharedPtr parameter_update_timer_;

    uint32_t next_object_id_;
    std::vector<TrackedObject> tracked_objects_;

    // Configuration parameters
    double max_object_width_;
    double min_object_width_;
    int min_object_points_;
    double detection_angle_deg_;
    double tracking_dist_threshold_;
    int tracking_staleness_threshold_;

    struct OrientedBoundingBox
    {
        geometry_msgs::msg::Point center;
        double width; // Corresponds to marker scale.x
        double depth; // Corresponds to marker scale.y
        tf2::Quaternion orientation;
    };

    // Helper function to check if a point is within the detection angle
    bool is_within_detection_angle(double x, double y, double detection_angle_deg) const
    {
        double angle_rad = std::atan2(y, x);
        double detection_angle_rad = detection_angle_deg * M_PI / 180.0;
        double min_angle_rad = -detection_angle_rad / 2.0;
        double max_angle_rad = detection_angle_rad / 2.0;
        return angle_rad >= min_angle_rad && angle_rad <= max_angle_rad;
    }

    // Check if object center is within detection boundary
    bool is_object_in_detection_zone(const Point2D &center, double detection_angle_deg) const
    {
        return is_within_detection_angle(center.x, center.y, detection_angle_deg);
    }

    // Check if object center is within tracking boundary (detection zone + buffer)
    bool is_object_in_tracking_zone(const Point2D &center, double detection_angle_deg, double buffer_distance) const
    {
        double detection_angle_rad = detection_angle_deg * M_PI / 180.0;
        double min_angle_rad = -detection_angle_rad / 2.0;
        double max_angle_rad = detection_angle_rad / 2.0;

        double object_distance = std::hypot(center.x, center.y);
        if (object_distance == 0.0)
            return true; // At origin

        // Calculate the angle offset needed for the buffer at this distance
        double angle_buffer = std::atan(buffer_distance / object_distance);

        double object_angle = std::atan2(center.y, center.x);
        return object_angle >= (min_angle_rad - angle_buffer) && object_angle <= (max_angle_rad + angle_buffer);
    }

    OrientedBoundingBox compute_oriented_bounding_box(const std::vector<geometry_msgs::msg::Point32> &points)
    {
        OrientedBoundingBox obb;
        if (points.size() < 2)
        {
            // Not enough points to define a box
            return obb;
        }

        // 1. Find the two farthest points to determine the primary axis (orientation)
        double max_dist_sq = 0.0;
        size_t p1_idx = 0, p2_idx = 0;
        for (size_t i = 0; i < points.size(); ++i)
        {
            for (size_t j = i + 1; j < points.size(); ++j)
            {
                double dist_sq = std::pow(points[i].x - points[j].x, 2) + std::pow(points[i].y - points[j].y, 2);
                if (dist_sq > max_dist_sq)
                {
                    max_dist_sq = dist_sq;
                    p1_idx = i;
                    p2_idx = j;
                }
            }
        }

        const auto &p1 = points[p1_idx];
        const auto &p2 = points[p2_idx];

        // 2. Calculate the orientation (yaw) of this primary axis
        double angle = std::atan2(p2.y - p1.y, p2.x - p1.x);
        obb.orientation.setRPY(0, 0, angle);

        // 3. Project all points onto the primary axis and its perpendicular to find extents
        double min_proj_axis = std::numeric_limits<double>::max();
        double max_proj_axis = std::numeric_limits<double>::lowest();
        double min_proj_perp = std::numeric_limits<double>::max();
        double max_proj_perp = std::numeric_limits<double>::lowest();

        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);

        for (const auto &p : points)
        {
            // Project point onto the primary axis
            double proj_axis = p.x * cos_a + p.y * sin_a;
            min_proj_axis = std::min(min_proj_axis, proj_axis);
            max_proj_axis = std::max(max_proj_axis, proj_axis);

            // Project point onto the perpendicular axis
            double proj_perp = -p.x * sin_a + p.y * cos_a;
            min_proj_perp = std::min(min_proj_perp, proj_perp);
            max_proj_perp = std::max(max_proj_perp, proj_perp);
        }

        // 4. Calculate dimensions and center point
        obb.width = max_proj_axis - min_proj_axis;
        obb.depth = max_proj_perp - min_proj_perp;

        // The center of the box in the rotated frame
        double center_proj_axis = (min_proj_axis + max_proj_axis) / 2.0;
        double center_proj_perp = (min_proj_perp + max_proj_perp) / 2.0;

        // Rotate the center back to the original frame
        obb.center.x = center_proj_axis * cos_a - center_proj_perp * sin_a;
        obb.center.y = center_proj_axis * sin_a + center_proj_perp * cos_a;
        obb.center.z = 0.0; // Assuming 2D

        // Add a small minimum size to prevent zero-scale markers
        obb.width = std::max(obb.width, 0.05);
        obb.depth = std::max(obb.depth, 0.05);

        return obb;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Use cached parameter values instead of reading each time
        double buffer_distance = max_object_width_;

        // Calculate tracking angle automatically based on buffer at minimum distance
        // Assume minimum meaningful distance of 0.5m for angle calculation
        double min_distance = 0.5;
        double angle_buffer_rad = std::atan(buffer_distance / min_distance);
        double angle_buffer_deg = angle_buffer_rad * 180.0 / M_PI;
        double tracking_angle_deg = std::min(180.0, detection_angle_deg_ + 2 * angle_buffer_deg);

        publish_fov_marker(msg->header, detection_angle_deg_, msg->range_max);
        publish_tracking_zone_marker(msg->header, detection_angle_deg_, buffer_distance, msg->range_max);

        // Use calculated tracking angle for initial scan processing
        auto points = scan_to_points(msg, tracking_angle_deg);
        auto clusters = group_points(points, max_object_width_);
        auto candidates = filter_clusters(clusters, min_object_points_, min_object_width_, max_object_width_);

        // Filter candidates to only those in tracking zone before updating tracking
        std::vector<ObjectCandidate> tracking_candidates;
        for (const auto &candidate : candidates)
        {
            if (is_object_in_tracking_zone(candidate.center, detection_angle_deg_, buffer_distance))
            {
                tracking_candidates.push_back(candidate);
            }
        }

        update_tracking(tracking_candidates, msg->header.stamp);

        // Only publish objects that are within the detection zone
        publish_objects(msg->header, detection_angle_deg_);
        publish_markers(msg->header, detection_angle_deg_);
    }

    std::vector<Point2D> scan_to_points(const sensor_msgs::msg::LaserScan::SharedPtr &msg, double angle_deg)
    {
        std::vector<Point2D> points;
        double angle_rad = angle_deg * M_PI / 180.0;
        double min_angle_rad = -angle_rad / 2.0;
        double max_angle_rad = angle_rad / 2.0;

        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            float scan_angle = msg->angle_min + i * msg->angle_increment;
            if (scan_angle >= min_angle_rad && scan_angle <= max_angle_rad)
            {
                float range = msg->ranges[i];
                if (std::isfinite(range))
                {
                    points.push_back({range * std::cos(scan_angle), range * std::sin(scan_angle)});
                }
            }
        }
        return points;
    }

    std::vector<std::vector<Point2D>> group_points(const std::vector<Point2D> &points, double max_dist_between_points)
    {
        std::vector<std::vector<Point2D>> groups;
        if (points.empty())
            return groups;

        std::vector<Point2D> current_group;
        current_group.push_back(points[0]);

        for (size_t i = 1; i < points.size(); ++i)
        {
            double distance = std::hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
            if (distance < max_dist_between_points)
            {
                current_group.push_back(points[i]);
            }
            else
            {
                if (!current_group.empty())
                    groups.push_back(current_group);
                current_group.clear();
                current_group.push_back(points[i]);
            }
        }
        if (!current_group.empty())
            groups.push_back(current_group);
        return groups;
    }

    std::vector<ObjectCandidate> filter_clusters(const std::vector<std::vector<Point2D>> &clusters, int min_points, double min_width, double max_width)
    {
        std::vector<ObjectCandidate> candidates;
        for (const auto &cluster : clusters)
        {
            if (cluster.size() < static_cast<size_t>(min_points))
                continue;
            double object_width = std::hypot(cluster.back().x - cluster.front().x, cluster.back().y - cluster.front().y);
            if (object_width < min_width || object_width > max_width)
                continue;

            // Check diameter constraint more thoroughly
            bool valid_diameter = true;
            for (size_t i = 0; i < cluster.size(); ++i)
            {
                for (size_t j = i + 1; j < cluster.size(); ++j)
                {
                    if (std::hypot(cluster[i].x - cluster[j].x, cluster[i].y - cluster[j].y) > max_width)
                    {
                        valid_diameter = false;
                        break;
                    }
                }
                if (!valid_diameter)
                    break;
            }

            if (valid_diameter)
            {
                ObjectCandidate candidate;
                candidate.points = cluster;

                // Calculate center and closest point distance
                double sum_x = 0, sum_y = 0;
                double min_dist_sq = std::numeric_limits<double>::max();
                for (const auto &p : cluster)
                {
                    sum_x += p.x;
                    sum_y += p.y;
                    min_dist_sq = std::min(min_dist_sq, p.x * p.x + p.y * p.y);
                }
                candidate.center = {sum_x / cluster.size(), sum_y / cluster.size()};
                candidate.closest_distance = std::sqrt(min_dist_sq);
                candidates.push_back(candidate);
            }
        }
        return candidates;
    }

    void update_tracking(const std::vector<ObjectCandidate> &candidates, const rclcpp::Time &stamp)
    {
        std::vector<bool> candidate_matched(candidates.size(), false);

        // Try to match existing tracks
        for (auto &track : tracked_objects_)
        {
            double min_dist = tracking_dist_threshold_;
            int best_candidate_idx = -1;
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (!candidate_matched[i])
                {
                    double dist = std::hypot(track.center_.x - candidates[i].center.x, track.center_.y - candidates[i].center.y);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best_candidate_idx = i;
                    }
                }
            }

            if (best_candidate_idx != -1)
            {
                track.update(candidates[best_candidate_idx], stamp);
                candidate_matched[best_candidate_idx] = true;
            }
            else
            {
                track.increment_staleness();
            }
        }

        // Remove stale tracks
        tracked_objects_.erase(
            std::remove_if(tracked_objects_.begin(), tracked_objects_.end(),
                           [&](const TrackedObject &track)
                           { return track.staleness_ > tracking_staleness_threshold_; }),
            tracked_objects_.end());

        // Add new tracks for unmatched candidates
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (!candidate_matched[i])
            {
                tracked_objects_.emplace_back(next_object_id_++, candidates[i], stamp);
            }
        }
    }

    void publish_objects(const std_msgs::msg::Header &header, double detection_angle_deg)
    {
        auto object_array_msg = std::make_unique<lidar_object_detection::msg::DetectedObjectArray>();
        object_array_msg->header = header;

        // First, filter and calculate distance to each tracked object's center for sorting
        std::vector<std::pair<double, TrackedObject *>> sorted_tracks;
        for (auto &track : tracked_objects_)
        {
            // Only publish objects that are within the detection zone
            if (is_object_in_detection_zone(track.center_, detection_angle_deg))
            {
                double dist = std::hypot(track.center_.x, track.center_.y);
                sorted_tracks.push_back({dist, &track});
            }
        }

        // Sort by distance (closest first)
        std::sort(sorted_tracks.begin(), sorted_tracks.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        for (const auto &pair : sorted_tracks)
        {
            const auto &track = *pair.second;
            lidar_object_detection::msg::DetectedObject obj_msg;
            obj_msg.header = header;
            obj_msg.object_id = track.object_id_;
            obj_msg.center.x = track.center_.x;
            obj_msg.center.y = track.center_.y;
            obj_msg.center.z = 0;
            obj_msg.velocity = track.velocity_;

            // Convert bounding box points
            for (const auto &p32 : track.bounding_box_)
            {
                obj_msg.bounding_box.points.push_back(p32);
            }
            object_array_msg->objects.push_back(obj_msg);
        }

        objects_pub_->publish(std::move(object_array_msg));
    }

    void publish_markers(const std_msgs::msg::Header &header, double detection_angle_deg)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        // Add a "delete all" marker to clear old visualizations from previous cycles
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header = header;
        clear_marker.id = 0;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        int marker_id = 1;
        for (const auto &track : tracked_objects_)
        {
            // Skip if there are no points to form a box
            if (track.bounding_box_.empty())
                continue;

            bool in_detection_zone = is_object_in_detection_zone(track.center_, detection_angle_deg);

            // --- Create a CUBE marker for the object's bounding box ---
            OrientedBoundingBox obb = compute_oriented_bounding_box(track.bounding_box_);
            visualization_msgs::msg::Marker box_marker;
            box_marker.header = header;
            box_marker.ns = "object_box";
            box_marker.id = marker_id++;
            box_marker.type = visualization_msgs::msg::Marker::CUBE;
            box_marker.action = visualization_msgs::msg::Marker::ADD;

            // Set the pose of the marker. This is the center of the OBB.
            box_marker.pose.position = obb.center;

            // **FIX 1: Use tf2::convert for explicit type conversion**
            // This is more robust than tf2::toMsg for avoiding template deduction errors.
            tf2::convert(obb.orientation, box_marker.pose.orientation);

            // Set the scale of the marker.
            box_marker.scale.x = obb.width;
            box_marker.scale.y = obb.depth;
            box_marker.scale.z = 0.1; // A fixed height for visualization

            // Set different colors for objects inside/outside detection zone
            box_marker.color.a = 0.7; // Make it semi-transparent
            if (in_detection_zone)
            {
                // Red for objects in detection zone (published)
                box_marker.color.r = 1.0;
                box_marker.color.g = 0.1;
                box_marker.color.b = 0.1;
            }
            else
            {
                // Blue for objects being tracked but outside detection zone
                box_marker.color.r = 0.1;
                box_marker.color.g = 0.1;
                box_marker.color.b = 1.0;
            }
            marker_array.markers.push_back(box_marker);

            // --- Create a TEXT_VIEW_FACING marker for ID and velocity ---
            double textSize = 0.09; // Text size in meters

            visualization_msgs::msg::Marker text_marker;
            text_marker.header = header;
            text_marker.ns = "object_info";
            text_marker.id = marker_id++;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position.x = track.center_.x;
            text_marker.pose.position.y = track.center_.y;
            text_marker.pose.position.z = 0.5; // Offset text above the box

            text_marker.scale.z = textSize; // Text height
            text_marker.color.a = 1.0;
            text_marker.color.r = 1.0;
            text_marker.color.g = 1.0;
            text_marker.color.b = 1.0;
            std::stringstream ss;
            ss.precision(2);
            ss << std::fixed << "ID: " << track.object_id_ << "\n"
               << "V: " << std::hypot(track.velocity_.x, track.velocity_.y) << " m/s";
            if (!in_detection_zone)
            {
                ss << "\n(TRACKING)";
            }
            text_marker.text = ss.str();
            marker_array.markers.push_back(text_marker);
        }

        marker_pub_->publish(marker_array);
    }

    /**
     * @brief Publishes a LINE_STRIP marker to visualize the detection field of view.
     * @param header The message header to use for the marker.
     * @param detection_angle_deg The detection angle in degrees.
     * @param range_max The maximum range to draw the lines.
     */
    void publish_fov_marker(const std_msgs::msg::Header &header, double detection_angle_deg, float range_max)
    {
        visualization_msgs::msg::Marker fov_marker;
        fov_marker.header = header;
        fov_marker.ns = "fov";
        fov_marker.id = 0;
        fov_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        fov_marker.action = visualization_msgs::msg::Marker::ADD;

        fov_marker.pose.orientation.w = 1.0;
        fov_marker.scale.x = 0.05; // Line width

        fov_marker.color.a = 0.8;
        fov_marker.color.r = 0.0;
        fov_marker.color.g = 1.0;
        fov_marker.color.b = 0.0;

        double detection_angle_rad = detection_angle_deg * M_PI / 180.0;
        double min_angle_rad = -detection_angle_rad / 2.0;
        double max_angle_rad = detection_angle_rad / 2.0;

        // Use range_max from scan, but cap it for visualization purposes if it's too large or inf
        float viz_range = std::isfinite(range_max) ? range_max : 15.0;

        geometry_msgs::msg::Point p_origin, p_min, p_max;
        p_origin.x = p_origin.y = p_origin.z = 0;

        p_min.x = viz_range * std::cos(min_angle_rad);
        p_min.y = viz_range * std::sin(min_angle_rad);
        p_min.z = 0;

        p_max.x = viz_range * std::cos(max_angle_rad);
        p_max.y = viz_range * std::sin(max_angle_rad);
        p_max.z = 0;

        fov_marker.points.push_back(p_min);
        fov_marker.points.push_back(p_origin);
        fov_marker.points.push_back(p_max);

        fov_marker_pub_->publish(fov_marker);
    }

    void publish_tracking_zone_marker(const std_msgs::msg::Header &header, double detection_angle_deg, double buffer_distance, float range_max)
    {
        visualization_msgs::msg::Marker tracking_marker;
        tracking_marker.header = header;
        tracking_marker.ns = "tracking_zone";
        tracking_marker.id = 1;
        tracking_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        tracking_marker.action = visualization_msgs::msg::Marker::ADD;

        tracking_marker.pose.orientation.w = 1.0;
        tracking_marker.scale.x = 0.03;

        tracking_marker.color.a = 0.6;
        tracking_marker.color.r = 0.0;
        tracking_marker.color.g = 0.5;
        tracking_marker.color.b = 1.0;

        double detection_angle_rad = detection_angle_deg * M_PI / 180.0;
        double min_angle_rad = -detection_angle_rad / 2.0;
        double max_angle_rad = detection_angle_rad / 2.0;

        float viz_range = std::isfinite(range_max) ? range_max : 15.0;

        // Create curved tracking zone boundaries
        std::vector<geometry_msgs::msg::Point> left_boundary, right_boundary;

        for (double distance = 0.5; distance <= viz_range; distance += 0.5)
        {
            double angle_buffer = std::atan(buffer_distance / distance);

            geometry_msgs::msg::Point left_point, right_point;
            left_point.x = distance * std::cos(min_angle_rad - angle_buffer);
            left_point.y = distance * std::sin(min_angle_rad - angle_buffer);
            left_point.z = 0;

            right_point.x = distance * std::cos(max_angle_rad + angle_buffer);
            right_point.y = distance * std::sin(max_angle_rad + angle_buffer);
            right_point.z = 0;

            left_boundary.push_back(left_point);
            right_boundary.push_back(right_point);
        }

        // Add left boundary
        for (const auto &point : left_boundary)
        {
            tracking_marker.points.push_back(point);
        }

        // Add right boundary (in reverse to complete the shape)
        for (auto it = right_boundary.rbegin(); it != right_boundary.rend(); ++it)
        {
            tracking_marker.points.push_back(*it);
        }

        // Close the shape by connecting back to start
        if (!left_boundary.empty())
        {
            tracking_marker.points.push_back(left_boundary.front());
        }

        fov_marker_pub_->publish(tracking_marker);
    }

    void update_config()
    {
        max_object_width_ = this->get_parameter(PARAM_MAX_OBJECT_WIDTH).as_double();
        min_object_width_ = this->get_parameter(PARAM_MIN_OBJECT_WIDTH).as_double();
        min_object_points_ = this->get_parameter(PARAM_MIN_OBJECT_POINTS).as_int();
        detection_angle_deg_ = this->get_parameter(PARAM_DETECTION_ANGLE).as_double();
        tracking_dist_threshold_ = this->get_parameter(PARAM_TRACKING_DIST_THRESHOLD).as_double();
        tracking_staleness_threshold_ = this->get_parameter(PARAM_TRACKING_STALENESS_THRESHOLD).as_int();

        // Parameter validation
        if (max_object_width_ <= 0.0)
        {
            RCLCPP_ERROR(this->get_logger(), "max_object_width_m must be positive");
            return;
        }
        if (min_object_width_ <= 0.0 || min_object_width_ >= max_object_width_)
        {
            RCLCPP_ERROR(this->get_logger(), "min_object_width_m must be positive and less than max_object_width_m");
            return;
        }
        if (min_object_points_ <= 0)
        {
            RCLCPP_ERROR(this->get_logger(), "min_amount_of_object_points must be positive");
            return;
        }
        if (detection_angle_deg_ <= 0.0 || detection_angle_deg_ > 360.0)
        {
            RCLCPP_ERROR(this->get_logger(), "detection_angle_deg must be between 0 and 360 degrees");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Configuration updated - Detection angle: %.1f°, Object width: %.2f-%.2fm",
                    detection_angle_deg_, min_object_width_, max_object_width_);
    }
};

// --- Parameter Constants Definition ---
const std::string LidarObjectDetectionNode::PARAM_MAX_OBJECT_WIDTH = "max_object_width_m";
const std::string LidarObjectDetectionNode::PARAM_MIN_OBJECT_WIDTH = "min_object_width_m";
const std::string LidarObjectDetectionNode::PARAM_MIN_OBJECT_POINTS = "min_amount_of_object_points";
const std::string LidarObjectDetectionNode::PARAM_DETECTION_ANGLE = "detection_angle_deg";
const std::string LidarObjectDetectionNode::PARAM_TRACKING_DIST_THRESHOLD = "tracking_dist_threshold_m";
const std::string LidarObjectDetectionNode::PARAM_TRACKING_STALENESS_THRESHOLD = "tracking_staleness_threshold";

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarObjectDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}