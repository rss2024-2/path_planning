import rclpy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive,AckermannDriveStamped
from geometry_msgs.msg import PoseArray,Point
from rclpy.node import Node
from visualization_msgs.msg import Marker

from .utils import LineTrajectory

from tf_transformations import euler_from_quaternion

import numpy as np


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1.0
        self.speed = 1.0
        self.wheelbase_length = 0.3

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.odom_sub = self.create_subscription(Odometry,self.odom_topic,self.pose_callback,1)

        self.lookahead_point_publisher=self.create_publisher(Marker,"/lookahead_point",1)
        
    def get_2Dpose_from_3Dpose(self, position, quaternion):

        yaw = euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])[2]

        return np.array([position.x,position.y,yaw])
    
    def plot_line(self, x, y, publisher, color = (1., 0., 0.), frame = "/base_link"):

        # Construct a line
        line_strip = Marker()
        line_strip.type = Marker.LINE_STRIP
        line_strip.header.frame_id = frame

        # Set the size and color
        line_strip.scale.x = 0.1
        line_strip.scale.y = 0.1
        line_strip.color.a = 1.
        line_strip.color.r = color[0]
        line_strip.color.g = color[1]
        line_strip.color.g = color[2]

        # Fill the line with the desired values
        for xi, yi in zip(x,y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)

    def pose_callback(self, odometry_msg):
        
        pose=self.get_2Dpose_from_3Dpose(odometry_msg.pose.pose.position,odometry_msg.pose.pose.orientation)

        p2=None

        for p1 in self.trajectory.points[:]:
            
            if p2==None:
                p2=p1
                continue

            sp1=np.array(p1)-pose[:2]
            sp2=np.array(p2)-pose[:2]
            d=sp2-sp1

            qa=d.dot(d)
            qb=2*sp1.dot(d)
            qc=sp1.dot(sp1)-self.lookahead**2

            if qb**2-4*qa*qc<0:
                p2=p1
                continue
            else:
                t=(-qb+np.sqrt(qb**2-4*qa*qc))/(2*qa)
                if t<0 or t>1:
                    p2=p1
                    continue
                intersection_point=sp1+d*t

                self.plot_line([pose[0],pose[0]+intersection_point[0]],[pose[1],pose[1]+intersection_point[1]],self.lookahead_point_publisher,frame='map')

                eta=np.arctan2(intersection_point[1],intersection_point[0])-pose[2]
                delta=np.arctan2(2*self.lookahead*np.sin(eta),self.wheelbase_length)

                header=Header()
                header.stamp=self.get_clock().now().to_msg()
                header.frame_id="base_link"
                drive=AckermannDrive()
                drive.steering_angle=delta
                drive.steering_angle_velocity=0.0
                drive.speed=self.speed
                drive.acceleration=0.0
                drive.jerk=0.0
                stamped_msg=AckermannDriveStamped()
                stamped_msg.header=header
                stamped_msg.drive=drive
                self.drive_pub.publish(stamped_msg)
                break




    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
