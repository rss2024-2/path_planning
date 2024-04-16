import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import numpy as np
import heapq

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.map = None
        self.start_point = None
        self.end_point = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, msg):
        self.map = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def pose_cb(self, pose):
        self.start_point = pose.pose.pose.position

    def goal_cb(self, msg):
        self.end_point = msg.pose.position
        if self.map is not None and self.start_point is not None:
            path = self.plan_path(self.start_point, self.end_point, self.map)
            self.publish_trajectory(path)

    def plan_path(self, start_point, end_point, map):
        # A* search algorithm
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

        def a_star_search(graph, start, goal):
            frontier = []
            heapq.heappush(frontier, (0, start))
            came_from = {}
            cost_so_far = {start: 0}

            while frontier:
                current_cost, current_node = heapq.heappop(frontier)

                if current_node == goal:
                    break

                for next_node in graph.neighbors(current_node):
                    new_cost = cost_so_far[current_node] + graph.cost(current_node, next_node)
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + heuristic(goal, next_node)
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current_node

            path = []
            current_node = goal
            while current_node != start:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path

        # Convert map to graph representation
        class Graph:
            def __init__(self, map):
                self.map = map

            def neighbors(self, node):
                neighbors = []
                x, y = node
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.map.shape[0] and 0 <= ny < self.map.shape[1] and self.map[nx][ny] == 0:
                            neighbors.append((nx, ny))
                return neighbors

            def cost(self, from_node, to_node):
                return heuristic(from_node, to_node)

        graph = Graph(map)
        start = (int(start_point.x), int(start_point.y))
        goal = (int(end_point.x), int(end_point.y))
        path = a_star_search(graph, start, goal)
        return path

    def publish_trajectory(self, path):
        trajectory_msg = PoseArray()
        trajectory_msg.header.frame_id = "map"
        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            trajectory_msg.poses.append(pose)
        self.traj_pub.publish(trajectory_msg)
        self.trajectory.publish_viz()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
