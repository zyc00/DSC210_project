import time
import numpy as np
import pyrealsense2 as rs
from realman.camera import base_camera


# import base_camera
# import cv2
# import open3d as o3d


# TODO: 加载配置文件

class RealSenseCamera(base_camera.BaseCamera):
    """Intel RealSense相机类"""

    def __init__(self, serial_num=None, color_resolution=[640, 480], depth_resolution=[640, 480], fps=6):
        """
        初始化相机对象
        :param _config: 相机配置参数，默认为空字典
        """
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        self._color_frames_rate = fps
        self._depth_frames_rate = fps
        self.timestamp = 0
        self.color_timestamp = 0
        self.depth_timestamp = 0
        self._colorizer = rs.colorizer()
        self._config = rs.config()
        self.camera_on = False
        self.serial_num = serial_num

    def _set__config(self):
        if self.serial_num != None:
            self._config.enable_device(self.serial_num)

        self._config.enable_stream(
            rs.stream.color,
            self._color_resolution[0],
            self._color_resolution[1],
            rs.format.rgb8,
            self._color_frames_rate,
        )
        self._config.enable_stream(
            rs.stream.depth,
            self._depth_resolution[0],
            self._depth_resolution[1],
            rs.format.z16,
            self._depth_frames_rate,
        )

    def start_camera(self):
        """
        启动相机并获取内参信息,如果后续调用帧对齐,则内参均为彩色内参
        """
        self._pipeline = rs.pipeline()
        self.point_cloud = rs.pointcloud()
        self._align = rs.align(rs.stream.color)
        self._set__config()
        self.profile = self._pipeline.start(self._config)

        self._depth_intrinsics = (
            self.profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._color_intrinsics = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.camera_on = True
        print(
            "color_resolution:",
            self._color_resolution,
            "depth_resolution:",
            self._depth_resolution,
        )
        print(
            "color_fps:",
            self._color_frames_rate,
            "depth_fps:",
            self._depth_frames_rate,
        )

    def stop_camera(self):
        """
        停止相机
        """
        self._pipeline.stop()
        self.camera_on = False

    def set_resolution(self, color_resolution, depth_resolution):
        self._color_resolution = color_resolution
        self._depth_resolution = depth_resolution
        print(
            "Optional color resolution:"
            "[320, 180] [320, 240] [424, 240] [640, 360] [640, 480]"
            "[848, 480] [960, 540] [1280, 720] [1920, 1080]"
        )
        print(
            "Optional depth resolution:"
            "[256, 144] [424, 240] [480, 270] [640, 360] [640, 400]"
            "[640, 480] [848, 100] [848, 480] [1280, 720] [1280, 800]"
        )

    def set_frame_rate(self, color_fps, depth_fps):
        self._color_frames_rate = color_fps
        self._depth_frames_rate = depth_fps
        print("Optional color fps: 6 15 30 60 ")
        print("Optional depth fps: 6 15 30 60 90 100 300")

    # TODO: 调节白平衡进行补偿
    # def set_exposure(self, exposure):

    def read_frame(self, is_colorized_depth=False, is_point_cloud=False):
        """
        读取一帧彩色图像和深度图像
        :return: 彩色图像和深度图像的NumPy数组
        """
        while not self.camera_on:
            time.sleep(0.5)
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if is_colorized_depth:
            colorized_depth = np.asanyarray(
                self._colorizer.colorize(self.depth_frame).get_data()
            )
        else:
            colorized_depth = None
        if is_point_cloud:
            points = self.point_cloud.calculate(self.depth_frame)
            point_cloud = np.asanyarray(points.get_vertices())
        else:
            point_cloud = None
            # 获取时间戳单位为ms，对齐后color时间戳 > depth = aligned，选择color
            self.color_timestamp = color_frame.get_timestamp()
            self.depth_timestamp = depth_frame.get_timestamp()

        return color_image, depth_image, colorized_depth, point_cloud

    def read_align_frame(self, is_colorized_depth=False, is_point_cloud=False):
        """
        读取一帧对齐的彩色图像和深度图像
        :return: 彩色图像和深度图像的NumPy数组
        """
        while not self.camera_on:
            time.sleep(0.5)
        try:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            aligned_color_frame = aligned_frames.get_color_frame()
            self._aligned_depth_frame = aligned_frames.get_depth_frame()

            color_image = np.asanyarray(aligned_color_frame.get_data())
            depth_image = np.asanyarray(self._aligned_depth_frame.get_data())
            if is_colorized_depth:
                colorized_depth = np.asanyarray(
                    self._colorizer.colorize(self._aligned_depth_frame).get_data()
                )
            else:
                colorized_depth = None

            if is_point_cloud:
                points = self.point_cloud.calculate(self._aligned_depth_frame)
                # pcd = np.asanyarray(points.get_vertices())
                pcd = np.array(points.get_vertices())
                # points_list = []

                # # 将元组数据转换为列表
                # for point in pcd:
                #     points_list.append([point[0], point[1], point[2]])

                # # 将列表转换为 NumPy 数组
                # point_cloud = np.array(points_list)
                point_cloud = pcd.reshape(-1, 3)
            else:
                point_cloud = None

            # 获取时间戳单位为ms，对齐后color时间戳 > depth = aligned，选择color
            self.timestamp = aligned_color_frame.get_timestamp()
            return color_image, depth_image, colorized_depth, point_cloud

        except Exception as e:
            if "Frame didn't arrive within 5000" in str(e):
                device = self.profile.get_device()
                device.hardware_reset()
                print("reset ")
                return None

    def get_camera_intrinsics(self):
        """
        获取彩色图像和深度图像的内参信息
        :return: 彩色图像和深度图像的内参信息
        """
        # 宽高：.width, .height; 焦距：.fx, .fy; 像素坐标：.ppx, .ppy; 畸变系数：.coeffs
        print(
            "Width and height: .width, .height; Focal length: .fx, .fy; \
            Pixel coordinates: .ppx, .ppy; Distortion coefficient: .coeffs"
        )
        return self._color_intrinsics, self._depth_intrinsics

    def get_3d_camera_coordinate(self, depth_pixel, align=False):
        """
        获取相机坐标系下的三维坐标
        :param pixel:深度像素坐标

        :return: 深度值和相机坐标
        """

        # 获取该像素点对应的深度

    def get_3d_camera_coordinate(self, depth_pixel, align=False):
        """
        获取深度相机坐标系下的三维坐标
        :param depth_pixel:深度像素坐标
        :param align: 是否对齐

        :return: 深度值和相机坐标
        """

        # 获取该像素点对应的深度
        distance = self._aligned_depth_frame.get_distance(
            depth_pixel[0], depth_pixel[1]
        )
        # 对齐后无论是深度还是彩色都是一样的，所以传深度像素还是彩色像素都可以。
        if align:
            camera_coordinate = rs.rs2_deproject_pixel_to_point(
                self._color_intrinsics, depth_pixel, distance
            )
        else:
            camera_coordinate = rs.rs2_deproject_pixel_to_point(
                self._depth_intrinsics, depth_pixel, distance
            )
        return distance, camera_coordinate


if __name__ == "__main__":
    camera = RealSenseCamera()
    camera.start_camera()
    while 1:
        result = camera.read_align_frame(False, False)
        if result is None:
            continue
        (
            color_image,
            depth_image,
            _,
            point_cloud,
        ) = result

        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera left", color_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    camera.stop_camera()
