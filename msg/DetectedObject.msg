# This message defines a single tracked object.

std_msgs/Header header

# A unique ID for the tracked object
uint32 object_id

# The geometric center of the object in the laser scan's frame
geometry_msgs/Point center

# The relative velocity of the object
geometry_msgs/Vector3 velocity

# A polygon representing the convex hull or bounding box of the object's points
geometry_msgs/Polygon bounding_box